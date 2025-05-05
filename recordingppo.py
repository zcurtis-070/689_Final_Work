import time
import imageio
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import math

# Register Environment TODO create package and install locally
gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:ballbeamEnv",
)
# load model
model = PPO.load("logs/best_model/best_model.zip")

# make environment
env = gym.make("gymnasium_env/BallBeamWorld-v0", render_mode="rgb_array")
# unwrap to access attributes
base = env.unwrapped
# reset environment
obs, _ = env.reset()

# set up video writer
dt = base.dt
writer = imageio.get_writer(
    "ppo_ballbeam_best_model_longer.mp4",
    fps=int(1/dt*10),
    codec="libx264",
)

# set up lists to record data
states  = []
actions = []
times   = []

# get base force for plotting
base = base.weight/2
t  = 0.0

# store initial state
states.append(obs.copy())
# store initial time
times.append(t)

# Loop unitl the episode is done
done = False
while not done:
    # get action
    action, _ = model.predict(obs, deterministic=True)
    actions.append(action.copy())

    # step env
    obs, reward, done, truncated, info = env.step(action)
    t += dt

    # record state & time
    states.append(obs.copy())
    times.append(t)

    # write video frame
    frame = env.render()
    writer.append_data(frame)

writer.close()
env.close()
print("Wrote video to ppo_ballbeam_adjusted.mp4")

# store as array for plotting
all_states = np.array(states)  
actions    = np.array(actions)  
times      = np.array(times)    

# pull states we want to plot
sel_states  = all_states[:, [0, 2, 4]]
state_names = ["y", "phi", "s"]
action_names = ["F1", "F2"]


action_times = times[1:]
n_states = len(state_names)
fig, axes = plt.subplots(n_states+1, 1, sharex=True, figsize=(10, 2*(n_states+1)))

# plot y
axes[0].axhline(10, color='yellow', linestyle='--', label="Goal y")
axes[0].plot(times, sel_states[:, 0], linestyle='-')
axes[0].set_ylabel(f"{state_names[0]} (m)")
axes[0].grid(True)

# plot phi
axes[1].axhline(35, color='red', linestyle='--', label="phi_threshold")
axes[1].axhline(-35, color='red', linestyle='--', label="phi_threshold")
axes[1].plot(times, 180/math.pi*sel_states[:, 1], linestyle='-')
axes[1].set_ylabel(f"{state_names[1]} (deg)")
axes[1].grid(True)

# plot s
axes[2].axhline(8/2, color='red', linestyle='--', label="s_threshold")
axes[2].axhline(8/2, color='red', linestyle='--', label="s_threshold")
axes[2].plot(times, sel_states[:, 2], linestyle='-')
axes[2].set_ylabel(f"{state_names[2]} (m)")
axes[2].grid(True)

# plot actions
axes[3].axhline(50, color='red', linestyle='--', label="F_threshold")
axes[3].axhline(-50, color='red', linestyle='--', label="F_threshold")
axes[3].plot(action_times, actions[:, 1]+base, linestyle='-',
                label=action_names[1], color='orange')


axes[3].plot(action_times, actions[:, 0]+base, linestyle='-',
                label=action_names[0], color='blue')


axes[3].set_ylabel("action (N)")
axes[3].set_xlabel("Time (s)")
axes[3].legend(loc="upper right")
axes[3].grid(True)

fig.suptitle("State & Action Trajectories Over Time", y=1.02)
plt.tight_layout()
plt.savefig("state_action_history.png", dpi=200)
plt.show()
print("Saved state_action_history.png")
