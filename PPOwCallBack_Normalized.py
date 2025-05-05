import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# —— Your custom log‐name for easy file management —— 
Log_Name = "PPO_BallBeam_lr1e-3_bs64_512steps"

# 1) Register your custom Gymnasium env
gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:ballbeamEnv",
    max_episode_steps=1200,
)

# 2) Prepare log directories
LOG_DIR         = "./logs/monitor/"
TENSORBOARD_DIR = f"./logs/tensorboard/"
PLOT_DIR        = f"./logs/plots/"
BEST_MODEL_DIR  = f"./logs/best_model/"
VECNORMS_PATH   = f"./logs/vecnormalize.pkl"

for d in (LOG_DIR, TENSORBOARD_DIR, PLOT_DIR, BEST_MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# 3) Build & normalize the training environment
train_env = make_vec_env(
    "gymnasium_env/BallBeamWorld-v0",
    n_envs=4,
    monitor_dir=LOG_DIR,
)
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
)

# 4) Build & normalize the evaluation environment (no stats updates)
eval_vec = DummyVecEnv([lambda: gym.make("gymnasium_env/BallBeamWorld-v0")])
eval_env = VecNormalize(
    eval_vec,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    training=False,
    gamma=0.99,
)
eval_env.stats_path = VECNORMS_PATH

# 5) Callback to collect loss & reward per episode and save rolling‐avg plots
class LossRewardCallback(BaseCallback):
    def __init__(self, save_dir: str, rolling_window: int = 100, verbose=0):
        super().__init__(verbose)
        self.save_dir       = save_dir
        self.rolling_window = rolling_window
        os.makedirs(self.save_dir, exist_ok=True)

        self.ep_count   = 0
        self.ep_idxs    = []
        self.ep_rewards = []
        self.loss_idxs  = []
        self.ep_losses  = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:  # Monitor signals episode end
                self.ep_count += 1
                r = info["episode"]["r"]
                self.ep_idxs.append(self.ep_count)
                self.ep_rewards.append(r)
                l = self.logger.name_to_value.get("train/loss")
                if l is not None:
                    self.loss_idxs.append(self.ep_count)
                    self.ep_losses.append(l)
        return True

    def _on_training_end(self) -> None:
        # Save VecNormalize stats
        self.model.get_env().save(VECNORMS_PATH)

        # — Loss plot —
        plt.figure(figsize=(8,4))
        plt.plot(self.loss_idxs, self.ep_losses, '-')
        plt.xlabel("Episode")
        plt.ylabel("PPO Loss")
        plt.title(f"{Log_Name}: Loss vs Episodes")
        plt.grid(True)
        fp = os.path.join(self.save_dir, f"{Log_Name}_loss_vs_episodes.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"Saved loss plot to {fp}")

        # — Reward + rolling avg plot —
        rewards  = np.array(self.ep_rewards)
        episodes = np.array(self.ep_idxs)
        plt.figure(figsize=(8,4))
        plt.plot(episodes, rewards, alpha=0.3, label="Raw return")

        if len(rewards) >= self.rolling_window:
            kernel = np.ones(self.rolling_window)/self.rolling_window
            ma = np.convolve(rewards, kernel, mode="valid")
            plt.plot(episodes[self.rolling_window-1:], ma,
                     color="C1", label=f"{self.rolling_window}-ep MA")

        mean_r = rewards.mean() if rewards.size else 0.0
        plt.hlines(mean_r, episodes[0], episodes[-1],
                   linestyles="--", colors="C2",
                   label=f"Mean={mean_r:.1f}")

        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title(f"{Log_Name}: Reward vs Episodes (smoothed)")
        plt.legend()
        plt.grid(True)
        fp = os.path.join(self.save_dir, f"{Log_Name}_reward_vs_episodes_ma.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"Saved reward plot to {fp}")

# 6) Instantiate callbacks
plot_cb = LossRewardCallback(save_dir=PLOT_DIR, rolling_window=100)
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=f"./logs/{Log_Name}_eval_logs/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

# 7) Build & train the PPO model
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    learning_rate=1e-3,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    tensorboard_log=TENSORBOARD_DIR,
)

model.learn(
    total_timesteps=800_000,
    tb_log_name=Log_Name,
    callback=[plot_cb, eval_cb],
)

# 8) Save final policy & VecNormalize stats
model.save(Log_Name)
train_env.save(VECNORMS_PATH)
print(f"Training complete. Model: {Log_Name}.zip, VecNorm stats: {VECNORMS_PATH}")
