import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


model = PPO.load("logs/best_model/best_model")

gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:ballbeamEnv",
    max_episode_steps=1200,
)


eval_env = gym.make("gymnasium_env/BallBeamWorld-v0")
eval_env = Monitor(eval_env) 

def success_rate(model, env, n_episodes=100):
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        # step until the episode ends
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
        if info.get("is_success", False):
            successes += 1
    return successes / n_episodes


rate = success_rate(model, eval_env, n_episodes=200)
print(f"Success rate over 200 episodes: {rate*100:.1f}%")
