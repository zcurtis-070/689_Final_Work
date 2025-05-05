from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class ballbeamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_width = 800  # The width of the PyGame window
        self.window_height = 512 # The height of the PyGame window
        self.m1 = 1.0 # Mass of the Ball
        self.m2 = 2.0 # Mass of the Beam
        self.g = 9.81 # Gravitational acceleration
        self.r = 1.0 # Radius of the Ball in Meters
        self.L = 8.0 # Length of the Beam in Meters
        self.I1 = 2.0/5.0 * self.m1 * self.r**2 # Moment of Inertia of the Ball
        self.I2 = 1.0/12.0 * self.m2 * self.L**2 # Moment of Inertia of the Beam
        self.dt = 1/10 # Time step in seconds
        self.t_limit = 40 # Time limit in seconds
        self.weight = (self.m1 + self.m2) * self.g # Weight of the Ball and Beam
        self.phi_threshold_degrees = 35 # Maximum angle of the beam
        self.s_threshold = self.L/2 # Maximum distance of the ball from the beam center
        self.Goal_y = 20
        high = np.array([ 
                        20., # y
                        20., # yd    
                        math.pi, # phi
                        10., # phid
                        self.L/2+2.0, # s
                        10., # sd
                        math.pi],# theta wrapped
                        dtype=np.float32
                             )

        self.observation_space = spaces.Box(-high,high, dtype=np.float32) # 5D vector with values between -high and high

        # Continuous action space: 2D vector with values between 0 and 50
        self.action_space = spaces.Box(
            low = np.array([-self.weight/2,-self.weight/2], dtype=np.float32),
            high = np.array([+self.weight/2,+self.weight/2], dtype=np.float32),
            dtype = np.float32
        )

    
        self.state = None
        self.old_y = None
        self.t = 0.0
        self.return_so_far = 0.0
        self._viewer = None
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = np.zeros(7, dtype=np.float32) # 7D vector with values between -high and high
        self.state[2] = self.np_random.uniform(-0.35, 0.35) # phi randomly starts between -0.5 and 0.5
        self.t = 0.0 # reset time
        self.return_so_far = 0.0 # reset return
        self.old_y = float(self.state[1])
        if self._viewer is not None:
            self._viewer.fill((0,0,0))
            pygame.display.flip()
        return self.state.astype(np.float32).copy(), {}

    def step(self, action):
        delta1, delta2 = np.clip(action, self.action_space.low, self.action_space.high) # Clip action to be between -self.weight/2 and +self.weight/2
        base = self.weight/2 # Base force
        F1 = base + delta1 # Force 1
        F2 = base + delta2 # Force 2
        self.F1, self.F2 = float(F1), float(F2) # Set forces

        self.state = self.rk4(self.state,self.dt,F1,F2) # Runge-Kutta 4th order method to update state
        y, yd, phi, phid, s, sd, theta = self.state # Unpack state
        self.t += self.dt # Update time
        # reward = 100*yd*self.dt
        reward = 10*(y - self.old_y) - abs(s)  
        self.old_y = y # update old y 
        # Set up termination conditions
        ball_off_beam = abs(s) > self.s_threshold # Ball is off the beam
        angle_out = abs(phi) > math.radians(self.phi_threshold_degrees) # Beam is out of bounds
        success = y >= self.Goal_y and not ball_off_beam  and not angle_out# Ball is in the goal and not off the beam
        timeout = self.t >= self.t_limit # Time limit reached

        # Set up reward function
        if success:
            reward += 500.0
        if ball_off_beam:
            reward -= 100.0
        if angle_out:
            reward -= 100.0
         # Update return
        self.return_so_far += reward
        # Termination condition
        terminated = bool(success or ball_off_beam or angle_out) 
        # Truncation condition
        truncated = bool(timeout and not terminated) 

        if self.render_mode == "human":
            self.render()

        info = {
            "is_success": bool(success),

        }
        return self.state.astype(np.float32).copy(), reward, terminated, truncated, info
    
    # Define matrix for mass i.e. everything that is multiplied by ddot terms
    def _mass_matrix(self, phi, s):
        c, sn = np.cos(phi), np.sin(phi)
        return np.array([
            [self.m1 + self.m2,         self.m1*(s*c - self.r*sn),  self.m1*sn],
            [self.m1*(s*c -self.r*sn),  self.I2 + self.m1*s**2,     0.0       ],
            [self.m1*sn,                0.0,                        self.m1 + self.I1/self.r**2]
            ])
    
    # Define matrix for everything else i.e. everything on the right hand side of the equation
    def _rhs(self, phi, s, phid, sd, F1, F2):
        c, sn = math.cos(phi), math.sin(phi)
        # Right hand side for all y terms
        vY = (F1+F2 - (self.m1+self.m2)*self.g
               - self.m1*(sn*sd + s*c*phid - self.r*sn*phid 
                          + 2*c*sd*phid - s*sn*phid**2 
                          - self.r*c*phid**2))
        # Right hand side for all phi terms
        vP = (0.5*self.L*c*(F1-F2) - self.m1*s*self.g*c 
              + self.m1*s*self.r*phid**2 - 2*self.m1*s*sd*phid)
        # Right hand side for all s terms
        vS = -self.m1*self.g*sn + self.m1*s*phid**2 
        # Return the right hand side
        return np.array([vY, vP, vS])
    
    # Get ddot equations
    def _qddot(self, phi, s, phid, sd, F1, F2):
        M = self._mass_matrix(phi, s)
        rhs = self._rhs(phi, s, phid, sd, F1, F2)
        return np.linalg.solve(M, rhs)
    
    def rk4(self, st, dt, F1, F2):

        # Get the current state
        def deriv(x):
            y, yd, phi, phid, s, sd, th = x
            ydd, phidd, sdd = self._qddot(phi, s, phid, sd, F1, F2)
            thetad = sd/self.r - phid
            return np.array([yd, ydd, phid, phidd, sd, sdd, thetad])
        
        # Runge-Kutta 4th order method
        k1 = dt * deriv(st)
        k2 = dt * deriv(st + 0.5 * k1 *dt)
        k3 = dt * deriv(st + 0.5 * k2 *dt)
        k4 = dt * deriv(st + dt*k3)
        
        # Update state using the Runge-Kutta method
        return st + dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    
    def render(self):
        
        if self._viewer is None:
            pygame.init()
            self._viewer = pygame.display.set_mode((self.window_width, self.window_height))
            self._font = pygame.font.Font(None, 24)

        PX_PER_M = self.window_height/15 # Pixels per meter
        surf = self._viewer
        surf.fill((20, 20, 25))
        y,_, phi, _, s, _, theta = self.state
        # Make camera follow the beam
        cam_x, cam_y = 0.0, y
        def to_px(x_m, y_m):
            return (int(self.window_width/2 + (x_m -cam_x)*PX_PER_M),
                    int(self.window_height/2 - (y_m - cam_y)*PX_PER_M))
        

        #Draw goal bar
        pygame.draw.line(surf,(230,200,40), to_px(-self.L/2, self.Goal_y), to_px(self.L/2, self.Goal_y), 4)
        
        # Draw the beam
        hx, hy = (self.L/2)*math.cos(phi), (self.L/2)*math.sin(phi)
        pL, pR =  to_px(-hx, y-hy), to_px(hx, y+hy)
        pygame.draw.line(surf, (40, 90, 190), pL, pR, 5)

        # Draw the ball
        bx = s*math.cos(phi) - self.r*math.sin(phi)
        by = y + s*math.sin(phi) + self.r*math.cos(phi)
        cx, cy  = to_px(bx, by)
        rad_px = int(self.r * PX_PER_M)
        pygame.draw.circle(surf, (180, 180, 180), (cx, cy), rad_px)

        # Draw Spoke
        alpha = theta + phi
        sx = cx + 0.8*rad_px*math.cos(alpha)
        sy = cy + 0.8*rad_px*math.sin(alpha)
        pygame.draw.line(surf, (255, 0, 0), (cx, cy), (sx, sy), 3)
        
        # Output some text of states
        txt1 =  f"y: {y:.2f} s: {s:.2f} "\
                f"F1: {self.F1:.2f} "\
                f"F2: {self.F2:.2f} "\
                f"phi: {math.degrees(phi):.2f}°"\
                f"time: {self.t:.2f}s"          
        surf.blit(self._font.render(txt1, True, (255, 255, 255)), (10, 10))
        pygame.display.flip()

        if self.render_mode == "human":
            # flip to the screen
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array":
            # return H×W×3 array (pygame’s is W×H×3 so we transpose)
            arr = pygame.surfarray.array3d(surf)
            return np.transpose(arr, (1,0,2))
        else:
            return None 
        
    def close(self):
        if self._viewer is not None:
            pygame.quit()
            self._viewer = None
