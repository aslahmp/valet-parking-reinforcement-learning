import gym
from gym import spaces
import numpy as np

class LAVPEnv(gym.Env):
    def __init__(self, grid_size=(10, 10), num_users=3):
        super(LAVPEnv, self).__init__()
     
        self.grid_size = grid_size
        self.num_users = num_users
        
        # action space 8 possible movements
        self.action_space = spaces.Discrete(8)
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
            4: (-1, -1), # top-left
            5: (-1, 1),  # top-right
            6: (1, -1),  # bottom-left
            7: (1, 1)    # bottom-right
        }
        
        # observation space agent position + pickup/drop-off spots
        self.observation_space = spaces.Box(
            low=0, high=max(grid_size), 
            shape=(2 + 2 * num_users,), dtype=np.int32
        )
        
        # start position and coordinate 
        self.start_pos = np.array([0, 0])  # starting position
        self.cp_pos = np.array([grid_size[0] - 1, grid_size[1] - 1])  # parking Point
        self.user_pickup = [np.random.randint(0, grid_size[0], 2) for _ in range(num_users)]
        self.user_dropoff = [np.random.randint(0, grid_size[0], 2) for _ in range(num_users)]
        
        # internal state
        self.agent_pos = np.copy(self.start_pos)
        self.users_served = [0] * num_users
        self.current_user = 0
        self.time_steps = 0

    # def get_spots(self):
    #     return np.array([self.start_pos, self.cp_pos, self.user_pickup, self.user_dropoff])
    def get_spots(self):
        spots = np.concatenate(
            [self.start_pos.reshape(1, -1), self.cp_pos.reshape(1, -1)] +
            [pickup.reshape(1, -1) for pickup in self.user_pickup] +
            [dropoff.reshape(1, -1) for dropoff in self.user_dropoff]
        )
        return spots
    def reset(self):

        self.agent_pos = np.copy(self.start_pos)
        self.users_served = [0] * self.num_users
        self.current_user = 0
        self.time_steps = 0
        
        return self._get_obs()

    def step(self, action):

        self.time_steps += 1
        
        # update agent position
        aget_old_pos = np.copy(self.agent_pos)
        move = self.actions[action]
        new_pos = self.agent_pos + move
        new_pos = np.clip(new_pos, [0, 0], np.array(self.grid_size) - 1)
        self.agent_pos = new_pos
        

        done = False
        
        if  self.users_served[self.current_user]==0:  # If user not yet served
            if np.array_equal(self.agent_pos, self.user_pickup[self.current_user]):
                # pickup point reached

                self.users_served[self.current_user] = 1
        elif self.users_served[self.current_user] == 1 and np.array_equal(self.agent_pos, self.user_dropoff[self.current_user]):
                # drop off point reached

                self.users_served[self.current_user] = 2
                self.current_user += 1 if self.current_user < self.num_users - 1 else 0

        # check if all users are served and AV reaches CP
        if all(status == 2 for status in self.users_served) and np.array_equal(self.agent_pos, self.cp_pos):
            done = True
        return self._get_obs(), 0, done, self._get_info(new_pos, aget_old_pos)

    def _get_obs(self):
   
        obs = np.concatenate([self.agent_pos] + self.user_pickup + self.user_dropoff)
        return obs
    def _get_info(self,current,previous):
        return {
            "distance": np.linalg.norm(
                current - previous
            )
        }
    def renderPath(self,path, mode='human'):

        grid = np.full(self.grid_size, ' . ', )
        

        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        

        
        for pickup in self.user_pickup:
            grid[pickup[0], pickup[1]] = 'P'
        for dropoff in self.user_dropoff:
            grid[dropoff[0], dropoff[1]] = 'D'    
     
        
        # parking point
        grid[self.cp_pos[0], self.cp_pos[1]] = 'C'
        for i, pos in enumerate(path):
            b=i+1


            grid[pos[0], pos[1]] = f'{b}'
        
        # Print the grid

        for row in grid:
            print("  ".join(row))
        print()
    def render(self, mode='human'):
        #prints to console
        grid = np.full(self.grid_size, '.', dtype=str)
        
        # agent position
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        
        # pickup 
        
        for pickup in self.user_pickup:
            grid[pickup[0], pickup[1]] = 'P'
        for dropoff in self.user_dropoff:
            grid[dropoff[0], dropoff[1]] = 'D'    
     
        
        # parking point
        grid[self.cp_pos[0], self.cp_pos[1]] = 'C'
        
        # Print the grid
        for row in grid:
            print(" ".join(row))
        print()