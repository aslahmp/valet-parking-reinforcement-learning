import numpy as np
import random

class AntColony1:
    def __init__(self, grid_size=16, alpha=2, beta=1, evaporation_rate=0.8, num_ants=50, max_iterations=100):
        self.grid_size = grid_size
        self.alpha = alpha  # pheromone influence
        self.beta = beta  # heuristic influence
        self.evaporation_rate = evaporation_rate
        self.num_ants = num_ants
        self.max_iterations = max_iterations

        self.pheromone_matrix = np.ones((grid_size, grid_size,grid_size, grid_size))
        

        self.actions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1),   # right
            (-1, -1),  # top-left
            (-1, 1),   # top-right
            (1, -1),   # bottom-left
            (1, 1),    # bottom-right
        ]

    def is_valid_move(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def heuristic(self, current, next_pos):

        return 1 / (np.linalg.norm(np.array(current) - np.array(next_pos)) + 1e-6)

    def update_pheromones(self, paths, distances):
   
        self.pheromone_matrix *= (1 - self.evaporation_rate)

      
        for path, distance in zip(paths, distances):
            pheromone_increase = 1 / distance
            for i in range(len(path)-1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]

                self.pheromone_matrix[x1, y1, x2, y2] += pheromone_increase
                self.pheromone_matrix[x2, y2, x1, y1] += pheromone_increase


    def find_shortest_path(self, start, end,env):



        best_path = None
        best_distance = float('inf')

        for iteration in range(self.max_iterations):
            paths = []
            distances = []

            for ant in range(self.num_ants):
                current_position = start
                path = [current_position]
                visited = set(path)

                while current_position != end:
                    x, y = current_position

                    # compute probabilities for valid moves
                    probabilities = []
                    next_positions = []
                    for dx, dy in self.actions:
                        nx, ny = x + dx, y + dy
                        if self.is_valid_move(nx, ny) and (nx, ny) not in visited:

                            next_positions.append((nx, ny))
                            tau = self.pheromone_matrix[x,y,nx, ny]
                            eta = self.heuristic(current_position, (nx, ny))

                            probabilities.append((tau ** self.alpha) * (eta ** self.beta))
                    probabilities_sum = sum(probabilities)
                    if not probabilities or probabilities_sum == 0 :  #stuck

                        # env.renderPath(path)
                        current_position = path[-2]
                        path.pop()
                        continue




                    
                    # if probabilities_sum == 0:
                    #     break

                    probabilities = [p / probabilities_sum for p in probabilities]

                    
                    next_position = random.choices(next_positions, probabilities)[0]
                    path.append(next_position)
                    visited.add(next_position)
                    current_position = next_position



                if current_position != end:
                    continue

                total_distance = sum(
                    np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
                    for i in range(len(path) - 1)
                )
                paths.append(path)
                distances.append(total_distance)

                # update best path
                if total_distance < best_distance:
                    best_path = path
                    best_distance = total_distance

            # update pheromone matrix
            self.update_pheromones(paths, distances)

        return best_path, best_distance