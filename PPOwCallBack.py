import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# — your custom name for easy file management —
Log_Name = "PPO_BallBeam_1e-3lr_128steps_64batch_0.99gamma_yenc_longer"

# 1) Register your custom Gymnasium env
gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:ballbeamEnv",
    max_episode_steps=1200,
)

# 2) Prepare log directories using Log_Name
LOG_DIR         = "./logs/monitor/"
TENSORBOARD_DIR = f"./logs/ensorboard/"
PLOT_DIR        = f"./logs/plots/"
BEST_MODEL_DIR  = f"./logs/best_model/"
EVAL_LOG_DIR    = f"./logs/eval_logs/"

for d in (LOG_DIR, TENSORBOARD_DIR, PLOT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR):
    os.makedirs(d, exist_ok=True)

# 3) Make a vectorized, Monitor-wrapped environment
env = make_vec_env(
    "gymnasium_env/BallBeamWorld-v0",
    n_envs=4,
    monitor_dir=LOG_DIR,
)

# 4) Callback to collect loss & reward per episode and save rolling-avg plots
class LossRewardCallback(BaseCallback):
    def __init__(self, save_dir: str, rolling_window: int = 100, verbose=0):
        super().__init__(verbose)
        self.save_dir       = save_dir
        self.rolling_window = rolling_window
        os.makedirs(self.save_dir, exist_ok=True)

        self.ep_count            = 0
        self.ep_idxs             = []
        self.ep_rewards          = []
        self.loss_idxs           = []
        self.ep_losses           = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
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
        # — Loss vs Episodes —
        plt.figure(figsize=(8,4))
        plt.plot(self.loss_idxs, self.ep_losses, '-')
        plt.xlabel("Episode")
        plt.ylabel("PPO Loss")
        plt.title(f"{Log_Name}: Loss vs Episodes")
        plt.grid(True)
        fp = os.path.join(self.save_dir, f"{Log_Name}_loss_vs_episodes.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"Saved loss plot to {fp}")

        # — Reward vs Episodes + Rolling Avg —
        rewards  = np.array(self.ep_rewards)
        episodes = np.array(self.ep_idxs)
        plt.figure(figsize=(8,4))
        plt.plot(episodes, rewards, alpha=0.3, label="Raw return")

        if len(rewards) >= self.rolling_window:
            kernel = np.ones(self.rolling_window) / self.rolling_window
            ma = np.convolve(rewards, kernel, mode='valid')
            ma_eps = episodes[self.rolling_window-1:]
            plt.plot(ma_eps, ma, color='C1',
                     label=f"{self.rolling_window}-ep MA")

        mean_r = rewards.mean() if rewards.size else 0.0
        plt.hlines(mean_r, episodes[0], episodes[-1],
                   linestyles='--', colors='C2',
                   label=f"Mean={mean_r:.1f}")

        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title(f"{Log_Name}: Reward vs Episodes (smoothed)")
        plt.legend()
        plt.grid(True)
        fp = os.path.join(self.save_dir, f"{Log_Name}_reward_vs_episodes_ma.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"Saved reward plot to {fp}")

# 5) Instantiate callbacks
plot_cb = LossRewardCallback(save_dir=PLOT_DIR, rolling_window=100)

eval_env = gym.make("gymnasium_env/BallBeamWorld-v0")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=EVAL_LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

# 6) Build and train the PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
    tensorboard_log=TENSORBOARD_DIR,
)

model.learn(
    total_timesteps=800_000,
    tb_log_name=Log_Name,
    callback=[plot_cb, eval_callback],
)

# 7) Save final model
model.save(Log_Name)
print(f"Training complete. Saved model as {Log_Name}.zip")
