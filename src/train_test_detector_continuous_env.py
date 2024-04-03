"""
This file loads a pregenerated training dataset and a pregenerated agent policy, and trains and tests an OOD detector.
"""

import os
import sys
import torch
import argparse
import time 
import hashlib
import random
import argparse
import gym
from collections import namedtuple
from typing import List, Optional, Union
from distutils.util import strtobool
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from detectors.PEDM.pedm_detector import PEDM_Detector
from detectors.DEXTER.DEXTER_detector import DEXTER_Detector
from detectors.CPD.CPD_Detector import CPD_Detector

import envs_continuous

def generate_time_id():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

#use the same utils as in Haider et al. (2023) for consistency across detectors
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

    parser.add_argument("--policy-path", type=str, default="../assets/policies/CartPole-v0/PPO_policy/5000000_timesteps/model.pth", help="the path to the policy to be loaded for rollout generation")
    parser.add_argument("--policy-name",default="PPO",choices=["PPO"],type=str, help="name/class of the policy that interacts with the env")
    parser.add_argument("--num-episodes", type=int, default=50, help="number of episodes to generate rollouts for")
    #currently not used
    parser.add_argument("--max-steps-per-episode", type=int, default=200,
        help="the number of steps to run in each environment per episode")
    parser.add_argument("--is-jit", type=int, default=1,
        help="if not zero then policy is jitted policy")
    parser.add_argument("--noise-mode", type=str, default="none",
                        help="state or obs or none")
    #environments
    parser.add_argument("--train-env-id", type=str, default= "MJCartpole-v0",
        help="name of the train environment")
    parser.add_argument("--test-env-id", type=str, default= "MJCartpole-v0",
        help="name of the test environment")
    parser.add_argument("--env-id", type=str, default="CartPole-v0",
        help="the id of the environment")
    parser.add_argument('--train-env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="set the correlations for noise in the train environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--test-env-noise-corr', type=parse_float_tuple, required=False, default=(0.0,0.0),help="set the correlations for noise in the test environment: (0.0) means no correlation of noise, (1.0, 0.0) means one-step correlation only, (0.0, 1.0) means two step correlation only, etc.")
    parser.add_argument('--train-noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to the train env")
    parser.add_argument('--test-noise-strength', type=float, required=False, default=1.0, help="the strength of the noise applied to the env")
    parser.add_argument("--train-injection-time", type=int, default=0,
        help="injection time of the anomaly (0 = from start of episode)")
     
    #detector-specific
    parser.add_argument("--detector-name", required=True, type=str, default = "PEDM_Detector", help="class/type of the detector to use")
    parser.add_argument("--detector-path", required=False, default = "", type=str, help="path to save the detector")
    parser.add_argument("--train-data-path", type=str, default="", help="Path of rollout data (to be used to train the detector)")

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

def make_env(env_id: str, 
             seed: int, 
             injection_time: int, 
             args):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    env = gym.make(env_id)
    env.set_seed(seed)
    kwargs = {}

    #the code is expecting to use train settings: so set injection time far away to the future
    #kwargs["injection_time"] = args.injection_time
    kwargs["injection_time"] = injection_time

    if args.noise_mode != "none":
        train_noise_coeffs = list(args.train_env_noise_corr)
        test_noise_coeffs = list(args.test_env_noise_corr)

        if args.noise_mode == "obs":
            kwargs["obs_noise_train"] = {"mod_corr_noise": train_noise_coeffs,
                                            "noise_strength": args.train_noise_strength}

            kwargs["obs_noise_test"] = {"mod_corr_noise": test_noise_coeffs,
                                            "noise_strength": args.test_noise_strength}

        elif args.noise_mode == "state":
            kwargs["state_noise_train"] = {"mod_corr_noise": train_noise_coeffs,
                                            "noise_strength": args.train_noise_strength}

            kwargs["state_noise_test"] = {"mod_corr_noise": test_noise_coeffs,
                                            "noise_strength": args.test_noise_strength}
        else:
            raise ValueError('Invalid noise mode: choose "obs" or "state"')

    env.set_noise(kwargs)
    env.reset()
    return env


#Define agent class
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class JITAgent():
    def __init__(self):
        super().__init__()
        return

    def load_from_path(self, path, device):
        self.actor = torch.jit.load(path, map_location=torch.device(device))

    def get_value(self, x):
        raise Exception("Not implemented!")

    def get_action_and_value(self, x, action=None):
        action = self.actor(x.float()[None])
        return action, None, None, None

#===BEGIN: IMPORTS FROM HAIDER ET AL 2023====#
#Source: https://github.com/FraunhoferIKS/pedm-ood/blob/main/oodd_runner.py
rollout = namedtuple("rollout", ["states", "actions", "rewards", "dones"])
def policy_rollout(env, policy, max_steps_per_episode):
    #create empty lists for states, actions, rewards, dones
    states, actions, rewards, dones = ([],[],[],[],)

    #reset the environment
    state = torch.Tensor(env.reset()).to(device)
    #store the state variable
    states.append(state.cpu().numpy())

    if hasattr(policy, "reset"):
        policy.reset()

    #run until done
    done = False
    while not done:
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(state)
            
            #take step
            next_state, reward, done, info = env.step(action.cpu().numpy())

            states.append(next_state)
            actions.append(action.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            
            next_state = torch.Tensor(next_state).to(device)
            done = torch.Tensor([done]).to(device)
            state = next_state

    #reshape arrays to conform to Haider et al. (2023)
    state_arr = np.array(states)
    state_arr = state_arr.reshape(state_arr.shape[0], -1)
    action_arr = np.array(actions).reshape(len(actions), -1)
    reward_arr = np.array(rewards).reshape(-1)
    done_arr = np.array(dones).reshape(-1)

    return state_arr, action_arr, reward_arr, done_arr

def n_policy_rollouts(env, 
                      policy, 
                      n_episodes, 
                      max_steps_per_episode,
                      verbose=False):
    episodes = []
    trange = tqdm(range(n_episodes), position=0, desc="episode", ncols=80) if verbose else range(n_episodes)
    for n in trange:
        states, actions, rewards, dones = policy_rollout(env, policy, max_steps_per_episode)
        episode = rollout(states, actions, rewards, dones)
        episodes.append(episode)

    return episodes

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
#===END: IMPORTS FROM HAIDER ET AL (2023)====#

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DEXTER_Detectors = ["DEXTER_Detector", "DEXTER_S_Detector"]
    CPD_Detectors = ["CPD_ocd", "CPD_Chan", "CPD_Mei", "CPD_XS"]

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

    #initialize the train environment
    #injection time set to 100000 to avoid any anomalies
    train_env = make_env(env_id = args.train_env_id,
                         seed = args.seed,
                         injection_time = 100000,
                         args=args)

    #Train the detector
    if args.detector_name in DEXTER_Detectors:
        #IMP: SPECIFY BATCH SIZE HERE
        #Batch size = window size
        n_dimensions = train_env.observation_space.shape[0]
        print("NUM DIMS: ", n_dimensions)

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

        patience = args.max_steps_per_episode

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

    #Test detector
    print("=== BEGIN TESTING DETECTOR ===")
    
    #initially: 0 test episodes completed
    test_episode_ctr = 0

    y_scores = []
    y_true = []
    ep_rewards_modified = []
    
    #LINAS: add for testing
    list_obs = []
    list_injection_times = []

    #initialize policy
    agent = None
    if os.path.exists(args.policy_path):
        if not args.is_jit:
            print("")
            print("=== LOAD POLICY ===")
            print("loading policy from:", args.policy_path)
            agent = Agent(train_env).to(device)
            agent.load_state_dict(torch.load(args.policy_path))
            agent.eval()
            print("=== POLICY LOADED ===")
        else:
            print("")
            print("=== LOAD JIT POLICY ===")
            agent = JITAgent()
            agent.load_from_path(args.policy_path, device)
            print("=== JIT POLICY LOADED ===")
    else:
        raise ValueError("the specified policy path does not exist! Please specify a proper policy path with --policy-path 'path_to_policy.pth'")

    while test_episode_ctr < args.num_test_episodes:
        #injection time = time at which we inject the anomaly/noise into the env
        #NOTE: Since it's only applied to Reacher, manually coded episde length (150)
        #If this script is expanded to other envs, episde length should be passed as an argument or accessed from the env
        injection_time = np.random.randint(5, 150 - 5)

        test_env = make_env(env_id = args.test_env_id, 
                            seed = args.seed,
                            injection_time = injection_time,
                            args=args)

        #get the rollouts
        obs, acts, rewards, dones = policy_rollout(env=test_env, 
                                                   policy=agent, 
                                                   max_steps_per_episode=args.max_steps_per_episode)
        ep_rewards_modified.append(np.sum(rewards))

        #Only calculate results if injection time is valid, ie comes before the end of episode (otheriwse, can't validate AUROC)
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
            anom_scores = train_test_detector_CPD(detector = detector,
                                                  args = args,
                                                  observations = obs,
                                                  actions = acts)

        else:
            anom_scores = detector.predict_scores(obs, acts)

        #get the real anomaly occurence vector
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

    #save results
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
    
    print("=======SAVING ANOM SCORES AND TRUE LABELS=======")
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