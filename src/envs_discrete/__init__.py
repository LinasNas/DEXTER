from gymnasium import register

register(id="IMANSCartpoleEnv-v0", 
         entry_point="envs_discrete.cartpole:IMANSCartpoleEnv",
         max_episode_steps=200)

register(id="IMANOCartpoleEnv-v0", 
         entry_point="envs_discrete.cartpole:IMANOCartpoleEnv",
         max_episode_steps=200)

register(id="TimeSeriesEnv-v0", 
         entry_point="envs_discrete.cartpole:TimeSeriesEnv",
         max_episode_steps=200)

register(id="IMANOAcrobotEnv-v0", 
         entry_point="envs_discrete.acrobot:IMANOAcrobotEnv",
         max_episode_steps=500)