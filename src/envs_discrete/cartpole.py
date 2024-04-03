"""
This file provides the environments with modified noise.

The purpose of this file is to be imported anywhere where the environments are needed.
"""

import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import logger, spaces
import math
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from typing import Optional
import pickle

#seed
np.random.seed(2023)

#AR Process
def normalized_autoregressive_process(coeffs, T):
    """
    :param coeffs: number of coeffs determines correlation cut-off
    :param T: length of timeseries
    :return:
    """

    coeffs = [-coeff for coeff in coeffs]
    ar1 = np.array([1] + list(coeffs)) 
    ma1 = np.array([1])
    MA_object1 = ArmaProcess(ar1, ma1)
    
    #normalize
    noise = MA_object1.generate_sample(nsample=T)
    normalized_noise = noise / np.std(noise)
    return normalized_noise

#IMANS = ARNS 
class IMANSCartpoleEnv(CartPoleEnv):
    """

    This environment noises the state transition dynamics with a noise vector
    that has been drawn from a AR Process at the beginning of the episode.
    NOTE: This environment is an MDP.
    """
    
    def __init__(self, 
                 mod_noise_std = tuple(0.025 * element for element in (0.25, 0.45, 0.06, 0.5)),
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(0.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):

        super().__init__(render_mode=render_mode)
        self.mod_noise_std = mod_noise_std

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength
        
        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time
        
        return

    def get_injection_time(self):
        return self.injection_time
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        retvals = super().reset(seed=seed, options=options)

        #generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]
        
        self.train_state_noise_vec = np.stack([normalized_autoregressive_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])
        self.test_state_noise_vec = np.stack([normalized_autoregressive_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0

        if self.injection_time == 0:
            state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength

            mod_retvals_0 = retvals[0] + state_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.test_noise_step_ctr += 1

        else:
            state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength * self.mod_noise_std

            mod_retvals_0 = retvals[0] + state_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.train_noise_step_ctr += 1

        return mod_retvals

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        #apply noise to state
        #if the environment is not yet in injection time, apply train noise
        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            #slice a 4 dimensional vector of noise
            state_noise = self.train_state_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength

            #apply noise to the state
            self.state = tuple(val + noise for val, noise in zip(self.state, state_noise))

            #track both counters
            self.train_noise_step_ctr += 1
            self.step_counter += 1
            
        else:
            #slice a 4 dimensional vector of test noise
            state_noise = self.test_state_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength

            #apply the test noise 
            self.state = tuple(val + noise for val, noise in zip(self.state, state_noise))

            #only need to track the test noise counter
            self.test_noise_step_ctr += 1
            self.step_counter += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0

        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

            self.train_noise_step_ctr = 0
            self.test_noise_step_ctr = 0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


#IMANO = ARNO 
class IMANOCartpoleEnv(CartPoleEnv):
    """
    This environment noises the observations with a noise vector
    that has been drawn from AR Process at the beginning of the episode.

    NOTE: This environment is a POMDP.
    """

    def __init__(self, 
                 #customize by st dev of each dimension
                 mod_noise_std = tuple(0.25 * element for element in (0.75, 0.5, 0.25, 0.4)),
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(0.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):
        
        super().__init__(render_mode=render_mode)
        self.mod_noise_std = mod_noise_std

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time

        return

    def get_injection_time(self):
        return self.injection_time
        
    def get_episode_length(self):
        return self.ep_length
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        retvals = super().reset(seed=seed, options=options)

        # generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]

        self.train_obs_noise_vec = np.stack([normalized_autoregressive_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])
        self.test_obs_noise_vec = np.stack([normalized_autoregressive_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0

        #if injection_time is 0, inject test noise throughout the episode
        if self.injection_time == 0:
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength * self.mod_noise_std 
            mod_retvals_0 = retvals[0] + obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 
            self.test_noise_step_ctr += 1

        else:
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength

            mod_retvals_0 = retvals[0] + obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 
            self.train_noise_step_ctr += 1

        return mod_retvals
    

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.mod_noise_std * self.train_noise_strength
            obs = tuple(val + noise for val, noise in zip(self.state, obs_noise))
            self.train_noise_step_ctr += 1
            self.step_counter += 1

        else:
            # When reached injection time: inject test noise
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.mod_noise_std * self.test_noise_strength
            obs = tuple(val + noise for val, noise in zip(self.state, obs_noise))
            self.test_noise_step_ctr += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0

        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

            self.train_noise_step_ctr = 0
            self.test_noise_step_ctr = 0
            
        if self.render_mode == "human":
            self.render()

        return np.array(obs, dtype=np.float32), reward, terminated, False, {}

#TimeSeriesEnv = ARTS
class TimeSeriesEnv(CartPoleEnv):
    """
    Time series environment that only returns noise. Noise is drawn from a AR Process at the beginning of the episode.
    The environment is built on CartPole-v0, so it could be used in the same train/test loop as other cartpole envs. 
    """

    def __init__(self, 
                 train_mod_corr_noise=(0.0,0.0), 
                 train_noise_strength=1.0, 
                 test_mod_corr_noise=(1.0,0.0),
                 test_noise_strength=1.0,
                 ep_length=200, 
                 render_mode: Optional[str] = None,
                 step_counter = 0,
                 injection_time = 0):
        
        super().__init__(render_mode=render_mode)

        self.train_mod_corr_noise = train_mod_corr_noise
        self.train_noise_strength = train_noise_strength
        self.test_mod_corr_noise = test_mod_corr_noise
        self.test_noise_strength = test_noise_strength

        self.ep_length = ep_length

        self.step_counter = step_counter
        self.injection_time = injection_time

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32
        )

        return
    
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        retvals = super().reset(seed=seed, options=options)

        # generate the noise variates for the whole episode in advance
        T = self.ep_length + 1
        obs_dim = self.observation_space.shape[0]
        self.train_obs_noise_vec = np.stack([normalized_autoregressive_process(self.train_mod_corr_noise, T) for _ in range(obs_dim)])
        self.test_obs_noise_vec = np.stack([normalized_autoregressive_process(self.test_mod_corr_noise, T) for _ in range(obs_dim)])
        
        self.train_noise_step_ctr = 0
        self.test_noise_step_ctr = 0
        self.step_counter = 0

        #if injection_time is 0, inject test noise throughout the episode
        if self.injection_time == 0:
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength

            mod_retvals_0 = obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.test_noise_step_ctr += 1

        else:
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength

            mod_retvals_0 = obs_noise
            mod_retvals_0 = np.array(mod_retvals_0, dtype = np.float32)
            mod_retvals = (mod_retvals_0, retvals[1]) 

            self.train_noise_step_ctr += 1
        
        return mod_retvals
    

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        #step simply returns noise: correlated or not
        if self.step_counter < self.injection_time:
            #Before reaching injection time: inject train noise
            obs_noise = self.train_obs_noise_vec[:, self.train_noise_step_ctr] * self.train_noise_strength
            self.state = obs_noise

            self.train_noise_step_ctr += 1
            self.step_counter += 1

        else:
            # When reached injection time: inject test noise
            obs_noise = self.test_obs_noise_vec[:, self.test_noise_step_ctr] * self.test_noise_strength
            self.state = obs_noise

            self.test_noise_step_ctr += 1
            self.step_counter += 1

        terminated = False
        if not terminated:
            reward = 1.0
        
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}