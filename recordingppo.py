import time
import imageio
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import math

gym.register(
    id="gymnasium_env/BallBeamWorld-v0",
    entry_point="gymnasium_env.envs:ballbeamEnv",
)
model = PPO.load("logs/best_model/best_model.zip")


env = gym.make("gymnasium_env/BallBeamWorld-v0", render_mode="rgb_array")
base = env.unwrapped
obs, _ = env.reset()


writer = imageio.get_writer(
    "ppo_ballbeam_deltay_minus_abs_s.mp4",
    fps=env.metadata["render_fps"],
    codec="libx264",
)


states  = []
actions = []
times   = []


dt = base.dt
base = base.weight/2
# L = base.r
t  = 0.0


states.append(obs.copy())
times.append(t)


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
    time.sleep(1/100)


writer.close()
env.close()
print("Wrote video to ppo_ballbeam_adjusted.mp4")


all_states = np.array(states)  
actions    = np.array(actions)  
times      = np.array(times)    


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
