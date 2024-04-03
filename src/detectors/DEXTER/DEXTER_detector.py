import os
import argparse
import numpy as np
import time 
import hashlib
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.feature_extraction import settings
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import pandas as pd
import pickle
import concurrent.futures
import random
from utils.data import (
    load_object,
    n_policy_rollouts,
    save_object,
    split_train_test_episodes,
)
from utils.stats import eval_metrics

def extract_features_from_multidim_batch(batch, settings):
    num_dimensions = batch.shape[1]
    all_features = []
    
    def extract_features_for_dim(dim):
        df = pd.DataFrame(batch[:, dim], columns=['value'])
        df['id'] = 0
        df['time'] = range(len(batch))

        X = extract_features(df, column_id="id", column_sort="time", column_value="value", 
                            impute_function=np.nanmean, default_fc_parameters=settings, 
                            disable_progressbar=True, n_jobs=1)
        return X.values[0]  # extract the row as a 1D array
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_features = list(executor.map(extract_features_for_dim, range(num_dimensions)))

    return np.hstack(all_features)

def preprocess_data(list_data, batch_size, step_size=1, sliding = False):
    processed_data = []

    if sliding == True:
        for episode in list_data:
            batched_array = [episode[i:i+batch_size] for i in range(0, len(episode) - batch_size + 1, step_size)]
            processed_data.append(batched_array)

    else:
        for episode in list_data:
            num_batches = episode.shape[0] // batch_size
            batched_array = [episode[i:i+batch_size] for i in range(0, num_batches * batch_size, batch_size)]
            processed_data.append(batched_array)

    return processed_data

class DEXTER_Detector:
    def __init__(self, 
                 n_dimensions,
                 batch_size,
                 sliding = False):

        self.n_dimensions = n_dimensions
        self.batch_size = batch_size

        self.detector = None
        self.imputer = None
        self.num_features_per_dim = None

    def train(
            self,
            args):
        """
        """

        #if extracted features available, load them
        if args.TF_train_data_feature_path != "" and args.TF_imputer_path != "":
            #load the extracted features      
            if os.path.exists(args.TF_train_data_feature_path):
                print("Loading feature extractions data from: ", args.TF_train_data_feature_path)

                with open(args.TF_train_data_feature_path, 'rb') as file:
                    features_imputed = pickle.load(file)    

            else:
                raise ValueError("the specified extrated train data feature path does not exist!")
            
            #load the impyter
            if os.path.exists(args.TF_imputer_path):
                print("Loading imputer from: ", args.TF_imputer_path)

                with open(args.TF_imputer_path, 'rb') as file:
                    imputer = pickle.load(file) 

            else:
                raise ValueError("the specified imputer path does not exist!")
            
        else:
            #load rollout data
            if os.path.exists(args.train_data_path):
                print("loading rollout data")
                ep_data = load_object(args.train_data_path)
            
            else:
                raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")
            
            train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)
            
            # initialize the detector
            print("")
            print("Extracting features from train data...")
            num_eps = len(train_ep_data)
            
            states_train = [ep.states for ep in train_ep_data]
            action_train = [ep.actions for ep in train_ep_data]

            processed_train_data = preprocess_data(list_data = states_train, 
                                                batch_size = self.batch_size,
                                                sliding = args.TF_sliding)

            settings_efficient = settings.EfficientFCParameters()

            features = []
            train_ep_ctr = 0
            batch_ctr = 0

            if args.TF_sliding == True:
                print("Using sliding window for feature extraction")
            else:
                print("Not using sliding window for feature extraction")

            for episode in processed_train_data:
                batch_ctr = 0
                print("Episode: ", train_ep_ctr)
                
                for batch in episode:
                    X = extract_features_from_multidim_batch(batch=batch, settings=settings_efficient)
                    features.append(X)
                    if args.TF_sliding == True:
                        if batch_ctr % 50 == 0:
                            print("Batch: ", batch_ctr)
                    else:
                        if batch_ctr % 5 == 0:
                            print("Batch: ", batch_ctr)
                    batch_ctr +=1 
                train_ep_ctr += 1
                
            features = np.vstack(features)

            # Impute missing values
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(features)
            # Transform the training data
            features_imputed = self.imputer.transform(features)

            print("TRAIN DATA PROCESSING FINISHED")

        #Train the detector
        ISOFOREST_MODELS = []
        
        if self.n_dimensions == 1:
            model = IsolationForest(random_state=2023)
            model.fit(features_imputed)
            ISOFOREST_MODELS.append(model)
            self.num_features_per_dim = features_imputed.shape[1] // self.n_dimensions 
            print("num_features_per_dim", self.num_features_per_dim)

        else:
            #One model per dimension
            self.num_features_per_dim = features_imputed.shape[1] // self.n_dimensions 
            for dim in range(self.n_dimensions):
                start_idx = dim * self.num_features_per_dim
                end_idx = (dim + 1) * self.num_features_per_dim
                features_imputed_dim = features_imputed[:, start_idx:end_idx]
                model = IsolationForest(random_state=2023)
                model.fit(features_imputed_dim)
                ISOFOREST_MODELS.append(model)

        self.detector = ISOFOREST_MODELS
        print("DETECTOR FITTED")


    def test(self, 
             args,
             observations, 
             actions):
        
        test_data = preprocess_data(list_data = [observations], 
                                    batch_size = self.batch_size, 
                                    sliding = args.TF_sliding)
        
        settings_efficient = settings.EfficientFCParameters()

        all_features_test = []

        for episode in test_data:
            batch_ctr = 0
                
            features_test = []
            
            for batch in episode:
                X = extract_features_from_multidim_batch(batch=batch, settings=settings_efficient)
                features_test.append(X)
                
                if args.TF_sliding == True:
                    if batch_ctr % 50 == 0:
                        print("Batch: ", batch_ctr)
                else:
                    if batch_ctr % 5 == 0:
                        print("Batch: ", batch_ctr)
                
                batch_ctr += 1
                
            features_test = np.vstack(features_test)
            
            # Impute missing values
            features_imputed_test = self.imputer.transform(features_test)
            all_features_test.append(features_imputed_test)

        print("TEST DATA PROCESSING FINISHED")

        for episode_idx, episode in enumerate(all_features_test):
            anom_scores = []

            #one model for each dim
            for i in range(episode.shape[0]):
                feats = episode[i,:]
                anomaly_scores_dim = []
                for dim in range(self.n_dimensions):
                    start_idx = dim * self.num_features_per_dim
                    end_idx = (dim + 1) * self.num_features_per_dim
                    feats_dim = feats[start_idx:end_idx].reshape(1, -1)
                    anomaly_scores_dim.append(-1 * self.detector[dim].decision_function(feats_dim)[0])
                anomaly_score = np.mean(anomaly_scores_dim)
                
                if args.TF_sliding == True:
                    #For the first 10 obs, add the same score (no sliding window)
                    if i == 0:
                        # Append the initial score for the first 10 steps
                        for _ in range(self.batch_size):
                            anom_scores.append(anomaly_score)
                    #if it's the last state, skip (to conform with other detectors)
                    elif i == episode.shape[0] - 1:
                        pass
                    else:
                        anom_scores.append(anomaly_score)

                else:
                    for _ in range(self.batch_size):
                        anom_scores.append(anomaly_score)

        return anom_scores