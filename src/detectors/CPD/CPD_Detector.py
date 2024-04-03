import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import os
from utils.data import (
    load_object,
    n_policy_rollouts,
    save_object,
    split_train_test_episodes,
)
from utils.stats import eval_metrics

ocd = importr("ocd")

import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

def float_vector(arr):
    if isinstance(arr, (int, float, np.int64, np.float32, np.float64)):
        arr = np.array([arr])
    else:
        pass
    return robjects.vectors.FloatVector(arr)

def arr_to_vec(arr):
    robjects.numpy2ri.activate()
    return numpy2ri.py2rpy(arr)

def dict_to_listvec(dic):
    robjects.numpy2ri.activate()
    return robjects.ListVector(dic)

def rnorm(n, mean = 0):
    return np.random.normal(loc = mean, size = int(n))

def c(*args):
    return np.concatenate(args)


class CPD_Detector:
    def __init__(self, 
                 args,
                 p, 
                 patience):

        self.thresh = "MC"
        self.MC_reps = 100
        self.p = p
        self.patience = patience

        if args.detector_name == "CPD_ocd":
            self.detector_name = "ocd"
        elif args.detector_name == "CPD_Chan":
            self.detector_name = "Chan"
        elif args.detector_name == "CPD_Mei":
            self.detector_name == "Mei"
        elif args.detector_name == "CPD_XS":
            self.detector_name == "XS"
        else:
            raise TypeError("No such type of CPD Detector")
        
    def train_test(
            self,
            args,
            observations,
            actions):
        
        self.detector = ocd.ChangepointDetector(dim=self.p, 
                                                method=self.detector_name,
                                                beta=1, 
                                                patience = self.patience,
                                                MC_reps = self.MC_reps,
                                                thresh=self.thresh)
        #LOAD train data
        if os.path.exists(args.train_data_path):
            print("loading rollout data")
            ep_data = load_object(args.train_data_path)   
        else:
            raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")

        train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)

        #==BEGIN TRAINING==#
        #needed because we have to reset the detector after each episode, by the implementation of it
        self.detector = ocd.setStatus(self.detector,'estimating')

        states_train = [ep.states for ep in train_ep_data]
        actions_train = [ep.actions for ep in train_ep_data]

        #for each episode:
        for episode_idx in range(0, len(train_ep_data)):
        
            #loop through the episode
            for i in range(0, len(actions_train[episode_idx])):
                
                #get the state
                state_train = states_train[episode_idx][i]
                action_train = actions_train[episode_idx][i]

                #If use actions:
                # x_new = np.concatenate((state_train, action_train), axis = 0)
                # x_new = float_vector(x_new)

                #else:
                if state_train.shape[0] == 1:
                    state_train = np.concatenate((state_train, state_train), axis = 0)
                else:
                    pass

                x_new = float_vector(state_train)

                self.detector = ocd.getData(self.detector, x_new)
        
        
        #TEST THE DETECTOR
        anom_scores = []
        anom_score = 0
        #iterate over each step in the episode
        self.detector = ocd.setStatus(self.detector, 'monitoring')
        for i in range(0, len(actions)):
            
            state = observations[i]
            action = actions[i]

            #If use actions:
            # x_new = np.concatenate((state, action), axis = 0)
            # x_new = float_vector(x_new)
            #ELSE
            x_new = float_vector(state)

            self.detector = ocd.getData(self.detector, x_new)
            status = ocd.status(self.detector)

            if isinstance(status[0], float) == True:
                anom_score = 1
            else:
                anom_score = 0

            anom_scores.append(anom_score)
        
        #reset the detector for the next episode
        self.detector = ocd.reset(self.detector)

        return anom_scores