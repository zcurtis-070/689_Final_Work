import gymnasium as gym
import numpy as np
gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:BallBeamEnv",
)

env = gym.make("gymnasium_env/BallBeamWorld-v0", render_mode="human")
obs, _ = env.reset()
done = False
while not done:
    act = np.array([env.weight/2 + env.np_random.uniform(-3, 3),
                    env.weight/2 + env.np_random.uniform(-3, 3)])
    obs, rew, term, trunk, _ = env.step(act)
    done =  term or trunk
env.close()