import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Log name for ease
Log_Name = "PPO_BallBeam_20m"

# Register Environment TODO create package and install locally
gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:ballbeamEnv",
    max_episode_steps=1200,
)

# Create directories for logging
LOG_DIR         = "./logs/monitor/"
TENSORBOARD_DIR = f"./logs/ensorboard/"
PLOT_DIR        = f"./logs/plots/"
BEST_MODEL_DIR  = f"./logs/best_model/"
EVAL_LOG_DIR    = f"./logs/eval_logs/"

for d in (LOG_DIR, TENSORBOARD_DIR, PLOT_DIR, BEST_MODEL_DIR, EVAL_LOG_DIR):
    os.makedirs(d, exist_ok=True)

# Create training environment
env = make_vec_env(
    "gymnasium_env/BallBeamWorld-v0",
    n_envs=4,
    monitor_dir=LOG_DIR,
)

# Custom callback to plot loss and reward
class LossRewardCallback(BaseCallback):
    def __init__(self, save_dir: str, rolling_window: int = 100, verbose=0):
        super().__init__(verbose)
        # Setup save directory
        self.save_dir       = save_dir
        # setup up rolling window size
        self.rolling_window = rolling_window
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        # Initialize lists to store episode data
        self.ep_count            = 0
        self.ep_idxs             = []
        self.ep_rewards          = []
        self.loss_idxs           = []
        self.ep_losses           = []

    # What happens at the end of each training step
    def _on_step(self) -> bool:
        # Check if the training step is at the end of an episode
        for info in self.locals.get("infos", []):
            # Check if the episode is done
            if "episode" in info:
                # Increment episode count and store rewards
                self.ep_count += 1
                # get reward from the info dict
                r = info["episode"]["r"]
                # count episodes
                self.ep_idxs.append(self.ep_count)
                # Store the episode reward
                self.ep_rewards.append(r)
                # get the training loss from the logger
                l = self.logger.name_to_value.get("train/loss")
                # save the loss if it exists
                if l is not None:
                    self.loss_idxs.append(self.ep_count)
                    self.ep_losses.append(l)
        return True

    # When training ends, plot the data
    def _on_training_end(self) -> None:
        # Plot loss vs episodes
        plt.figure(figsize=(8,4))
        plt.plot(self.loss_idxs, self.ep_losses, '-')
        plt.xlabel("Episode")
        plt.ylabel("PPO Loss")
        plt.title(f"{Log_Name}: Loss vs Episodes")
        plt.grid(True)
        fp = os.path.join(self.save_dir, f"{Log_Name}_loss_vs_episodes.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"Saved loss plot to {fp}")

        # create array of rewards and episode indices
        rewards  = np.array(self.ep_rewards)
        episodes = np.array(self.ep_idxs)
        # plot raw return
        plt.figure(figsize=(8,4))
        plt.plot(episodes, rewards, alpha=0.3, label="Raw return")

        # plot smoothed return
        if len(rewards) >= self.rolling_window:
            kernel = np.ones(self.rolling_window) / self.rolling_window
            ma = np.convolve(rewards, kernel, mode='valid')
            ma_eps = episodes[self.rolling_window-1:]
            plt.plot(ma_eps, ma, color='C1',
                     label=f"{self.rolling_window}-ep MA")

        # plot mean return
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

# use the custom callback
plot_cb = LossRewardCallback(save_dir=PLOT_DIR, rolling_window=100)

# setup evaluation environment
eval_env = gym.make("gymnasium_env/BallBeamWorld-v0")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=EVAL_LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

# setup PPO model
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

# train the model
model.learn(
    total_timesteps=800_000,
    tb_log_name=Log_Name,
    callback=[plot_cb, eval_callback],
)

# Save final model. Not necassary, but useful for later
model.save(Log_Name)
print(f"Training complete. Saved model as {Log_Name}.zip")
