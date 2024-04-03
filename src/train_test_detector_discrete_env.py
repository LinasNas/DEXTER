"""
This file loads a pregenerated training dataset and a pregenerated agent policy, and trains and tests an OOD detector.
"""
import sys
import os
import argparse
import numpy as np
import time 
import hashlib
import random
import argparse
from collections import namedtuple
from typing import List, Optional, Union
from distutils.util import strtobool
from datetime import datetime
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from detectors.PEDM.pedm_detector import PEDM_Detector
from detectors.DEXTER.DEXTER_detector import DEXTER_Detector
from detectors.CPD.CPD_Detector import CPD_Detector

import envs_discrete

def generate_time_id():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

#use the same utils as in Haider et al. 2023 for consistency across detectors
from utils.data import (
    load_object,
    n_policy_rollouts,
    save_object,
    split_train_test_episodes,
)
from utils.stats import eval_metrics

def generate_hash():
    timestamp = str(time.time())  # Use current timestamp as the identifier
    hash_object = hashlib.md5(timestamp.encode())  # 
    return hash_object.hexdigest()  # Get the hexadecimal representation of the hash

def parse_float_tuple(string):
    if string == "" or string == "None":
        return None
    try:
        # Remove parentheses and split the values using commas
        cleaned_string = string.strip('()')
        
        # Check if the cleaned string is empty
        if cleaned_string:
            floats = [float(x) for x in cleaned_string.split(',') if x.strip() != '']
            
            # Handle single-element tuple
            if len(floats) == 1:
                return floats[0],  # Add a trailing comma to create a single-element tuple
            else:
                return tuple(floats)
        else:
            raise ValueError("Empty string")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float value encountered")
    
def parse_str_or_int(string):
    if string == "random":
        return "random"
    else:
        return int(string)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="seed of the experiment")

    parser.add_argument("--policy-path", type=str, default="", help="the path to the policy to be loaded for rollout generation of both train and test environments")
    parser.add_argument("--train-data-path", type=str, default="", help="Path of rollout data (to be used to train the detector)")

    #environments
    parser.add_argument("--train-env-id", type=str, default="CartPole-v0",
        help="name of the train environment")
    parser.add_argument("--test-env-id", type=str, default="CartPole-v0",
        help="name of the test environment")
    parser.add_argument("--max-steps-per-episode", type=int, default=200,
        help="the number of steps to run in each environment per episode")
    
    parser.add_argument('--train-env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="set the correlations for noise in the train environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--train-noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to the train env")
    parser.add_argument('--test-env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="set the correlations for noise in the test environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--test-noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to the env")
    parser.add_argument("--train-injection-time", type=int, default=0,
        help="injection time of the anomaly (0 = from start of episode)")

    #detector-specific
    parser.add_argument("--detector-name", required=True, type=str, default = "PEDM_Detector", help="class/type of the detector to use")
    parser.add_argument("--detector-path", required=False, default = "", type=str, help="path to save the detector")

    #experiment-specific
    parser.add_argument("--experiment-id", required=False, type=str, default = "", help="experiment ID to use")

    #leave steps at 2000 to have the same number as Haider et al. (2023)
    parser.add_argument("--num-train-episodes", default=2000, type=int, help="number of training episodes to train the dynamics model")
    parser.add_argument("--num-test-episodes", default=100, type=int, help="number of episodes to test the detector")

    parser.add_argument("--num-envs", type=int, default=1,
        help="number of environments created in parallel")
    parser.add_argument("--capture-video", type=bool, default=False,
        help="Whether to create a video recording of the agent's trajectory")

    #DEXTER specific
    parser.add_argument("--TF-train-data-feature-path", type=str, default="", help="Path to feature extractions of train data")
    parser.add_argument("--TF-imputer-path", type=str, default="", help="Path to feature extractions of train data")
    parser.add_argument("--TF-sliding", type=lambda x: bool(strtobool(x)), default=False, help = "if toggled, use sliding window for feature extraction")

    args = parser.parse_args()
    return args


#===BEGIN: IMPORTS FROM CLEANRL====# 
#Source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def make_env(env_id, run_name = generate_hash(), capture_video = False, seed = None,  options = None):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env.unwrapped.ep_length = env.spec.max_episode_steps
    if options is not None:
        for k, v in options.items():
            setattr(env.unwrapped, k, v)
    return env
#===END: IMPORTS FROM CLEANRL====#

#===BEGIN: IMPORTS FROM HAIDER ET AL 2023====#
#Source: https://github.com/FraunhoferIKS/pedm-ood/blob/main/oodd_runner.py
rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])
def policy_rollout(env, policy):
    #create empty lists for states, actions, rewards, dones
    states, actions, rewards, dones = ([],[],[],[],)

    #reset the environment
    state = torch.Tensor(env.reset()[0]).to(device)
    states.append(state.cpu().numpy())
    
    done = False
    while not done:
            #pick action
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(state)
            
            #take step
            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            
            #calculate done
            done = terminated or truncated 

            #store
            states.append(next_state)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            
            state = torch.Tensor(next_state).to(device)

    #reshape arrays to conform with Haider et al 2023
    state_arr = np.array(states)
    state_arr = state_arr.reshape(state_arr.shape[0], -1)
    action_arr = np.array(actions).reshape(len(actions), -1)
    reward_arr = np.array(rewards).reshape(-1)
    done_arr = np.array(dones).reshape(-1)

    return state_arr, action_arr, reward_arr, done_arr


#Procedure to train Probabilistic Dynamics Ensemble Model (PEDM)
#Adopted from Haider et al. (2023)
#Source: https://github.com/FraunhoferIKS/pedm-ood/blob/main/oodd_runner.py
def train_detector_PEDM(
    args,
    env,
    num_train_episodes: int,
    detector_class: type,
    data_path: str = "",
    data: list = [],
    detector_kwargs: Optional[dict] = {},
    detector_fit_kwargs: Optional[dict] = {},
):
    """
    main function to train an env dynamics model with data from some policy in some env

    Args:
        env: env to collect experience in
        data_path: path to save to or load experience buffer from (if applicable)
        n_train_episodes: how many episodes to collect/train THE DETECTOR
        detector_name: type/class of the detector
        detector_kwargs: kwargs to pass for the detector constructor
        detector_fit_kwargs="kwargs for the training loop of the detector"

    Returns:
        detector: the trained ood detector
    """

    #load rollout data
    if os.path.exists(data_path):
        print("loading rollout data")
        ep_data = load_object(data_path)
    
    else:
        raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")

    train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)

    # initialize the detector
    detector = detector_class(env=env, 
                              normalize_data=True,
                              num_train_episodes=args.num_train_episodes)
    
    print("")
    print("Detector: ", args.detector_name)
    print("Training environment: ", args.train_env_id)
    print("")
    
    #train the detector
    detector.fit(train_ep_data=train_ep_data, val_ep_data=val_ep_data, **detector_fit_kwargs)
    print("")

    return detector
#===END: IMPORTS FROM HAIDER ET AL 2023====#

def train_test_detector_CPD(detector,
                            args, 
                            observations,
                            actions):
    
    anom_scores = detector.train_test(args = args, 
                                      observations = observations,
                                      actions = actions)
    
    return anom_scores

def train_detector_DEXTER(
    detector,
    args):

    detector.train(args)

    return detector


def test_detector_DEXTER(
    detector,
    args,
    observations, 
    actions):
    
    anom_scores = detector.test(args=args,
                                observations=observations,
                                actions=actions)

    return anom_scores

#===MAIN LOOP====#
if __name__ == "__main__":
    args = parse_args()
    print("seed: ", args.seed)

    if args.experiment_id == "":
        experiment_id = generate_time_id()
    else:
        experiment_id = args.experiment_id
    
    print("Experiment ID: ", experiment_id)

    #set seeds
    if args.seed != None:
        #torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        #np
        np.random.seed(args.seed)
        random.seed(args.seed)

        #R
        # robjects.r('set.seed(' + str(args.seed) + ')')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CPD_Detectors = ["CPD_ocd", "CPD_Chan", "CPD_Mei", "CPD_XS"]
    DEXTER_Detectors = ["DEXTER_Detector", "DEXTER_S_Detector"]

    if args.detector_name not in CPD_Detectors and args.detector_name not in DEXTER_Detectors:
        detector_class = globals()[args.detector_name]
        print("Testing on one of the detectors from Haider et al (2023)")

    elif (args.detector_name in CPD_Detectors):
        print("Testing on one of the CPD detectors")

    elif (args.detector_name in DEXTER_Detectors):
        print("Testing on one of the DEXTER detectors")
        if args.detector_name == "DEXTER_S_Detector":
            print("Using sliding window for feature extraction")
            args.TF_sliding = True
        else:
            print("Not using sliding window for feature extraction")
            args.TF_sliding = False

    else:
        print("Detector not found!")
    
    print("")
    print("DEVICE: ", device)
    print("")

    # initialize the train environment
    # train_injection_time = 0 by default, so the whole episode runs on the test params.
    # therefore, test params here should be the params of the train env of the detector
    options_train = {
        "test_mod_corr_noise": args.train_env_noise_corr,
        "test_noise_strength": args.train_noise_strength,
        "injection_time": args.train_injection_time}
    
    train_env = make_env(env_id = args.train_env_id, options = options_train, seed = args.seed)

    #TRAIN DETECTOR
    if args.detector_name in DEXTER_Detectors:
        #IMP: SPECIFY BATCH (WINDOW SIZE) HERE
        #By default, it's 10
        n_dimensions = train_env.observation_space.shape[0]
        print("num dims: ", n_dimensions)

        batch_size = 10

        #initialize
        detector = DEXTER_Detector(n_dimensions = n_dimensions, 
                                   batch_size=batch_size, 
                                   sliding = False)

        #train
        detector = train_detector_DEXTER(detector = detector,
                                         args = args)
        
        print("=== DEXTER DETECTOR TRAINED! ===\n")

    #If it's a CPD Detector, we have to train in with each test episode, so we skip training for now
    #else: train it now
    elif args.detector_name in CPD_Detectors:
        print("")
        print("DETECTOR TYPE: CPD")
        p = train_env.observation_space.shape[0]
        print("P: ", p)
        #ocd cannot deal with 1-dim obs, set to 2 and duplicate the obs
        if p == 1:
            p = 2
        else:
            pass
        print("P: ", p)

        if args.test_noise_strength <= 1.5:
            patience = args.max_steps_per_episode

        elif 1.5 < args.test_noise_strength <= 5.0:
            patience = 150

        else:
            patience = 100

        thresh = "MC"
        MC_reps = 100
        detector = CPD_Detector(args = args, 
                                p = p, 
                                patience = patience)
    
    else:
        print("")
        print("DETECTOR TYPE: PEDM")
        print(" ")
        print("=== BEGIN TRAINING DETECTOR ===")
        detector = train_detector_PEDM(
            args,
            env = train_env,
            num_train_episodes = args.num_train_episodes,
            data_path = args.train_data_path,
            detector_class = detector_class)
        print("=== DETECTOR TRAINED! ===\n")

        # Save the detector
        if not os.path.exists(os.path.dirname(args.detector_path)):
            args.detector_path = os.path.join(
                "data",
                "detectors",
                args.train_env_id,
                args.detector_name,
                f"{args.num_train_episodes}_ep",
                generate_hash(),
                "model.pth")
            
        parent_dir = os.path.dirname(os.getcwd())
        full_data_path = os.path.join(parent_dir, args.detector_path)
        os.makedirs(os.path.dirname(full_data_path), exist_ok=True)
        detector.save(full_data_path)
        print("")
        print("=== DETECTOR SAVED! ===\n")

    #TEST DETECTOR
    print("=== BEGIN TESTING DETECTOR ===")
    
    #initially: 0 test episodes completed
    test_episode_ctr = 0

    y_scores = []
    y_true = []
    ep_rewards_modified = []

    #initialize the policy
    #train_env is used here to make sure the dimensions of train & test envs are the same
    agent = Agent(train_env).to(device)

    #load the same policy on both agents
    if os.path.exists(args.policy_path):
        print("")
        print("=== LOAD POLICY ===")
        print("loading policy from:", args.policy_path)
        agent.load_state_dict(torch.load(args.policy_path))
        agent.eval()
        print("=== POLICY LOADED ===")
              
    else:
        raise ValueError("the specified policy path does not exist! Please specify a proper policy path with --policy-path 'path_to_policy.pth'")

    while test_episode_ctr < args.num_test_episodes:
        # injection time = time at which we inject the anomaly into the env
        # in cartpole, stronger noise significantly shortens the episode length
        # therefore, adjust injection time
        if args.test_env_id in ["IMANOCartpoleEnv-v0", "IMANSCartpoleEnv-v0"]:
            if args.test_noise_strength <= 1.5:
                injection_time = np.random.randint(5, train_env.ep_length - 5)
            elif 1.5 < args.test_noise_strength <= 5.0:
                injection_time = np.random.randint(5, (train_env.ep_length - 50) - 5)
            else:
                injection_time = np.random.randint(5, (train_env.ep_length - 100) - 5)
        
        else:
            injection_time = np.random.randint(5, train_env.ep_length - 5)

        #if the environment is the same for train and testing
        if args.train_env_noise_corr == args.test_env_noise_corr and args.train_noise_strength == args.test_noise_strength:
            options_test = {
                "train_mod_corr_noise": args.train_env_noise_corr,
                "train_noise_strength": args.train_noise_strength,
                "test_mod_corr_noise": args.test_env_noise_corr,
                "test_noise_strength": args.test_noise_strength,
                "injection_time": 0}
        else:
            options_test = {
                "train_mod_corr_noise": args.train_env_noise_corr,
                "train_noise_strength": args.train_noise_strength,
                "test_mod_corr_noise": args.test_env_noise_corr,
                "test_noise_strength": args.test_noise_strength,
                "injection_time": injection_time}

        test_env = make_env(env_id = args.test_env_id, options = options_test, seed = args.seed)

        #get the rollouts
        obs, acts, rewards, dones = policy_rollout(env=test_env, policy=agent)
        ep_rewards_modified.append(np.sum(rewards))

        #Only calculate results if injection time is valid, i.e. comes before the end of episode
        #Otherwise, can't calculate AUROC
        if (len(obs) - injection_time < 5):
            continue
        test_episode_ctr += 1


        #Generate anomaly scores
        if args.detector_name in DEXTER_Detectors:
            anom_scores = test_detector_DEXTER(
                detector = detector, 
                args = args,
                observations = obs, 
                actions = acts)
            
        elif args.detector_name in CPD_Detectors:
            print("DETECTOR", detector)
            anom_scores = train_test_detector_CPD(detector = detector,
                                                  args = args,
                                                  observations = obs,
                                                  actions = acts)

        else:
            anom_scores = detector.predict_scores(obs, acts)

        
        #get the true anomaly occurence vector
        anom_occurrence = [0 if i < injection_time else 1 for i in range(len(anom_scores))]

        print("Episode: ", test_episode_ctr)
        print("Anomaly injection time: ", injection_time)
        print("Length of episode: ", len(obs))
        print("")

        y_scores.extend(anom_scores)
        y_true.extend(anom_occurrence)

    #Calculate results, take larger of AUROC values
    auroc = eval_metrics(y_scores, y_true)
    results_dict = {
        "reward": round(np.mean(ep_rewards_modified), 2),
        "auroc": round(max(auroc, 1 - auroc), 2)}

    #Save results
    env_configuration = "env_noise_corr_" + "_".join([str(elem).replace(".", "p") for elem in args.test_env_noise_corr]) + "_noise_strength_" + str(args.test_noise_strength).replace('.', 'p')

    y_scores_path = os.path.join("../experiment_results",
                                 experiment_id,
                                 args.detector_name,
                                 args.test_env_id,
                                 env_configuration,
                                 f"{args.num_test_episodes}_ep",
                                 "y_scores.pkl")

    y_true_path = os.path.join("../experiment_results",
                               experiment_id,
                               args.detector_name,
                               args.test_env_id,
                               env_configuration,
                               f"{args.num_test_episodes}_ep",
                               "y_true.pkl")
    
    print("=======SAVING ANOMALY SCORES AND TRUE LABELS=======")
    save_object(y_scores, y_scores_path)
    save_object(y_true, y_true_path)
    
    print("=== TESTING COMPLETE ===")
    print("Detector:", args.detector_name)
    print("train_env", args.train_env_id)
    print("test_env", args.test_env_id)
    print("num train episodes:", args.num_train_episodes)
    print("num test episodes:", args.num_test_episodes)
    print("train_env_noise_corr", args.train_env_noise_corr)
    print("train_noise_strength", args.train_noise_strength)
    print("test_env_noise_corr", args.test_env_noise_corr)
    print("test_noise_strength", args.test_noise_strength)
    print("")
    print("RESULTS:")
    print(results_dict)

    results_path = os.path.join("../experiment_results",
                                    experiment_id,
                                    args.detector_name,
                                    args.test_env_id,
                                    env_configuration,
                                    f"{args.num_test_episodes}_ep",
                                    "results.txt")
    
    with open(results_path, 'w') as f:
        f.write(f"Detector: {args.detector_name}\n")
        f.write(f"train_env: {args.train_env_id}\n")
        f.write(f"test_env: {args.test_env_id}\n")
        f.write(f"num train episodes: {args.num_train_episodes}\n")
        f.write(f"num test episodes: {args.num_test_episodes}\n")
        f.write(f"train_env_noise_corr: {args.train_env_noise_corr}\n")
        f.write(f"train_noise_strength: {args.train_noise_strength}\n")
        f.write(f"test_env_noise_corr: {args.test_env_noise_corr}\n")
        f.write(f"test_noise_strength: {args.test_noise_strength}\n")
        f.write("\n")

        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")