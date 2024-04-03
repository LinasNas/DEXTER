# DEXTER

Repository for the paper "Rethinking Out-of-Distribution Detection for Reinforcement Learning: Advancing Methods for Evaluation and Detection" (AAAMAS 2024). 

The appendix of the paper can be found in the root directory of the repository as ```appendix.pdf```.

# Setup
The code is structured so that most of the experiments could be run on both CPUs and GPUs (DEXTER and CPDs also tend to run faster on CPUs). However, experiments on continuous action space environments -- namely Reacher -- use Mujoco, so running the code on GPUs is recommended in these cases.

When using CPUs, the main steps are as follows:
1. install ```requirements.txt```. 
2. To run CPD algorithms, make sure to have a [working version of R](https://posit.co/download/rstudio-desktop/) installed. You will not need use R, as it will be called in the back-end. Moreover, the ```ocd``` package is required, which can be installed by running ```install.packages("ocd")``` in R. 

When using GPUs, a docker container is recommended, and ```Dockerfile``` is provided in the repository. If you are new to docker containers, helper scripts are provided below:
1. run ```./docker_build.sh```. This will install the necessary packages specified in ```Dockerfile```, including a working version of mujoco and R.
2. run ```./docker_run.sh "GPU_array" /bin/bash```, such as ```./docker_run.sh "0,1" /bin/bash```to run on 0,1 GPUs.

# 1. Train and test OOD detectors 
All code for training and testing the detectors can be found under ```src```. The main scripts are:
1. ```train_test_detector_continuous_env.py``` 
2. ```train_test_detector_continuous_env.py```

These scripts are split by whether the environment has discrete action space (ARTS, ARNS/ARNO Cartpole, ARNS/ARNO Acrobot) or continuous action space (ARNS/ARNO Reacher), since the architecture of the agent navigating the environment will be different. 

The repository includes fully trained agent policies and rollouts for all environments and noise levels. This ensures that the agents have been trained on the appropriate noise level, so that during test-time only the noise structure in the environment is changed. 

## Training and testing OOD detectors on discrete action space environments
### General structure of the code
For all discrete environments, all detectors and all noise levels, the general command is the following:

```
python train_test_detector_discrete_env.py \
    --detector-name " " \
    --train-env-id " " \
    --test-env-id " " \
    --train-data-path " " \
    --policy-path  " " \
    --train-noise-strength  " " \
    --train-env-noise-corr " " \
    --test-noise-strength " " \
    --test-env-noise-corr " " \
    --num-train-episodes " " \
    --num-test-episodes " " \
    --experiment-id " " \
    --seed 2023
```

Arguments:
- ```detector-name```: Name of the detector to use. Options: ```[DEXTER_Detector, PEDM_Detector, CPD_ocd, CPD_ocd, CPD_Chan, CPD_Mei, CPD_XS]
- ```train-env-id```: Environment ID for training. Options: ```["TimeSeriesEnv-v0", "IMANOAcrobotEnv-v0", "IMANSAcrobotEnv-v0", "IMANOCartpoleEnv-v0", "IMANSCartpoleEnv-v0"]```, where ```"TimeSeriesEnv-v0"``` stands for ARTS environment, ```IMANO``` environments stand for ARNO environments, and ```IMANS``` environments stand for ARNS environments.
- ```test-env-id```: Environment ID for testing. Same options as for ```train-env-id```
- ```train-data-path```: Path to the training data, which should be a pickle file containing the rollouts, denoted as ```ep_data.pkl```
- ```policy-path```: Path to the policy that generated the rollouts, which should be specified in ```.pth``` format. 
- ```train-noise-strength```: Noise strength for training. 
- ```train-env-noise-corr```: Noise correlation used in training. It should always be specified as ```"(0.0,0.0)"``` for uncorrelated noise. 
- ```test-noise-strength```: Noise strength for testing. Has to be the same as ```train-noise-strength```.
- ```test-env-noise-corr```: Noise correlation used in testing. Options are: ```"(0.0,0.0)"``` for uncorrelated noise, ```"(0.95,)"``` for 1-step autocorrelation, and ```"(0.0, 0.95)"``` for 2-step autocorrelation. 
- ```num-train-episodes```: Number of training episodes. For ```DEXTER_Detector``` and CPD detectors, use ```1```. For ```PEDM_Detector```, use ```2000``` to abide by the training procedure in the original paper.
- ```num-test-episodes```: Number of testing episodes. For reproducability, use ```100```. 
- ```experiment-id```: ID of the experiment, e.g. ```"2023_12_02_12_00_00"```
- ```seed```: use  ```2023``` for the most accurate reproducability of the results. 

Example command:

To test the DEXTER Detector on ARNO Cartpole, with noise strength of 1.5 (corresponding to Light Noise), 1-step noise correlation for testing, use the following command:

```
python train_test_detector_discrete_env.py \
    --detector-name "DEXTER_Detector" \
    --train-env-id "IMANOCartpoleEnv-v0" \
    --test-env-id "IMANOCartpoleEnv-v0" \
    --train-data-path "../assets/rollouts/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p5/50_ep/ep_data.pkl" \
    --policy-path "../assets/policies/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p5/PPO_policy/500000_timesteps/model.pth" \
    --train-noise-strength 1.5 \
    --train-env-noise-corr "(0.0,0.0)" \
    --test-noise-strength 1.5 \
    --test-env-noise-corr "(0.95,)" \
    --num-train-episodes 1 \
    --num-test-episodes 100 \
    --experiment-id "2023_12_01_12_00_00" \
    --seed 2023
```

### Specific commands for each environment
All agent policies and rollouts are provided in this repository. Below you can find the relevant arguments that should be used in each case, for both 1-step and 2-step noise correlations:

| env name      | env-id                | noise level | noise-strength | train-data-path                                                                                                                                      | policy-path                                                                                                                                                    |
|---------------|-----------------------|-------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ARTS | TimeSeriesEnv-v0   | Universal       | 1.0            | ../assets/rollouts/TimeSeriesEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p0/50_ep/ep_data.pkl                                                  | ../assets/policies/TimeSeriesEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p0/PPO_policy/500000_timesteps/model.pth    
| ARNO Cartpole | IMANOCartpoleEnv-v0   | Light       | 1.5            | ../assets/rollouts/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p5/50_ep/ep_data.pkl                                                  | ../assets/policies/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p5/PPO_policy/500000_timesteps/model.pth                                         |
| ARNO Cartpole | IMANOCartpoleEnv-v0   | Medium      | 5.0            | ../assets/rollouts/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_5p0/50_ep/ep_data.pkl                                                  | ../assets/policies/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_5p0/PPO_policy/500000_timesteps/model.pth                                         |
| ARNO Cartpole | IMANOCartpoleEnv-v0   | Strong      | 7.0            | ../assets/rollouts/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_7p0/50_ep/ep_data.pkl                                                  | ../assets/policies/IMANOCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_7p0/PPO_policy/500000_timesteps/model.pth                                         |
| ARNS Cartpole | IMANSCartpoleEnv-v0   | Light       | 1.0            | ../assets/rollouts/IMANSCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p0/50_ep/ep_data.pkl                                                  | ../assets/policies/IMANSCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p0/PPO_policy/500000_timesteps/model.pth                                         |
| ARNS Cartpole | IMANSCartpoleEnv-v0   | Medium      | 10.0           | ../assets/rollouts/IMANSCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_10p0/50_ep/ep_data.pkl                                                 | ../assets/policies/IMANSCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_10p0/PPO_policy/500000_timesteps/model.pth                                        |
| ARNS Cartpole | IMANSCartpoleEnv-v0   | Strong      | 17.5           | ../assets/rollouts/IMANSCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_17p5/50_ep/ep_data.pkl                                                 | ../assets/policies/IMANSCartpoleEnv-v0/env_noise_corr_0p0_0p0_noise_strength_17p5/PPO_policy/500000_timesteps/model.pth                                        |
| ARNO Acrobot  | IMANOAcrobotEnv-v0    | Light       | 0.3            | ../assets/rollouts/IMANOAcrobotEnv-v0/env_noise_corr_0p0_0p0_noise_strength_0p3/50_ep/ep_data.pkl                                                   | ../assets/policies/IMANOAcrobotEnv-v0/env_noise_corr_0p0_0p0_noise_strength_0p3/PPO_policy/500000_timesteps/model.pth                                          |
| ARNO Acrobot  | IMANOAcrobotEnv-v0    | Medium      | 1.25           | ../assets/rollouts/IMANOAcrobotEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p25/50_ep/ep_data.pkl                                                  | ../assets/policies/IMANOAcrobotEnv-v0/env_noise_corr_0p0_0p0_noise_strength_1p25/PPO_policy/500000_timesteps/model.pth                                         |
| ARNO Acrobot  | IMANOAcrobotEnv-v0    | Strong      | 2.0            | ../assets/rollouts/IMANOAcrobotEnv-v0/env_noise_corr_0p0_0p0_noise_strength_2p0/50_ep/ep_data.pkl                                                    | ../assets/policies/IMANOAcrobotEnv-v0/env_noise_corr_0p0_0p0_noise_strength_2p0/PPO_policy/500000_timesteps/model.pth                                          |


## Training and testing OOD detectors on continuous action space environments
### General structure of the code
For continuous environments (namely Reacher), the code is almost exactly the same, with the only difference being that the ```env-id``` is always ```MJReacher-v0```, and additional argument ```--noise-mode``` is used to specified whether the noise is applied to the state (ARNS) or the observations (ARNO). 

The general command is as follows:
```
python train_test_detector_discrete_env.py \
    --detector-name " " \
    --train-env-id " " \
    --test-env-id " " \
    --train-data-path " " \
    --policy-path  " " \
    --train-noise-strength  " " \
    --train-env-noise-corr " " \
    --test-noise-strength " " \
    --test-env-noise-corr " " \
    --noise-mode " " \
    --num-train-episodes " " \
    --num-test-episodes " " \
    --experiment-id " " \
    --seed 2023
```

New arguments:
- ```noise-mode```: Noise mode for the environment. Options are: ```["obs", "state"]```, where ```"obs"``` stands for ARNO environments, and ```"state"``` stands for ARNS environments.

Example command:

To test the PEDM Detector on ARNO Reacher, with noise strength of 0.4 (corresponding to Light Noise), 2-step noise correlation for testing, use the following command:

```
python train_test_detector_continuous_env.py \
    --detector-name "PEDM_Detector" \
    --train-env-id "MJReacher-v0" \
    --test-env-id "MJReacher-v0" \
    --train-data-path "../assets/rollouts/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p4/50_ep/ep_data.pkl" \
    --policy-path "../assets/policies/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p4/best_model.zip_jit.pt" \
    --train-env-noise-corr "(0.0,0.0)" \
    --test-env-noise-corr "(0.95,)" \
    --train-noise-strength 0.4 \
    --test-noise-strength 0.4 \
    --noise-mode "obs" \
    --num-train-episodes 2000 \
    --num-test-episodes 100 \
    --experiment-id "2023_12_01_12_00_00" \
    --seed 2023
```

### Specific commands for each environment
Similarly, below you can find the relevant arguments that should be used in each case, for both 1-step and 2-step noise correlations:

| env name      | env-id       | noise mode | noise level | noise-strength | train-data-path                                                                                                                                      | policy-path                                                                                                                                                 |
|---------------|--------------|------------|-------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ARNO Reacher  | MJReacher-v0 | "obs"      | Light       | 0.4            | ../assets/rollouts/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p4/50_ep/ep_data.pkl                                                       | ../assets/policies/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p4/best_model.zip_jit.pt                                                           |
| ARNO Reacher  | MJReacher-v0 | "obs"      | Medium      | 0.6            | ../assets/rollouts/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p6/50_ep/ep_data.pkl                                                       | ../assets/policies/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p6/best_model.zip_jit.pt                                                           |
| ARNO Reacher  | MJReacher-v0 | "obs"      | Strong      | 2.0            | ../assets/rollouts/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_2p0/50_ep/ep_data.pkl                                                       | ../assets/policies/IMANOReacher-v0/env_noise_corr_0p0_0p0_noise_strength_2p0/best_model.zip_jit.pt                                                           |
| ARNS Reacher  | MJReacher-v0 | "state"    | Light       | 0.1            | ../assets/rollouts/IMANSReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p1/50_ep/ep_data.pkl                                                       | ../assets/policies/IMANSReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p1/best_model.zip_jit.pt                                                           |
| ARNS Reacher  | MJReacher-v0 | "state"    | Medium      | 0.15           | ../assets/rollouts/IMANSReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p15/50_ep/ep_data.pkl                                                      | ../assets/policies/IMANSReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p15/best_model.zip_jit.pt                                                          |
| ARNS Reacher  | MJReacher-v0 | "state"    | Strong      | 0.17           | ../assets/rollouts/IMANSReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p17/50_ep/ep_data.pkl                                                      | ../assets/policies/IMANSReacher-v0/env_noise_corr_0p0_0p0_noise_strength_0p17/best_model.zip_jit.pt                                                          |


# 2. Evaluate OOD detectors on CUSUM scripts
In the ```CUSUM``` directory, you can also find the script ```evalute_results.py```, which implements the decision rule for when to classify a test-time deployment as OOD that is described in the paper.

# Notes 
To maximize compatability with methodology from previous work, environments with continuous action spaces are constructed with ```gym``` and using ```mujoco```, while environments with discrete action spaces use ```gymnasium```. 
