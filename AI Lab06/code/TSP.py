import random

import numpy as np

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class TSP:
    def __init__(self, input_X=None, input_Y=None, input_M=None, input_Alpha=None, input_Beta=None, input_Pho=None):
        if input_X is None:
            self.X = [1304, 3639, 4177, 3712, 3488, 3326, 3238, 4196, 4312, 4386, 3007, 2562, 2788, 2381, 1332, 3715,
                      3918, 4061, 3780, 3676, 4029, 4263, 3429, 3507, 3394, 3439, 2935, 3140, 2545, 2778, 2370]
        if input_Y is None:
            self.Y = [2312, 1315, 2244, 1399, 1535, 1556, 1229, 1004,  790,  570, 1970, 1756, 1491, 1676,  695, 1678,
                      2179, 2370, 2212, 2578, 2838, 2931, 1908, 2367, 2643, 3201, 3240, 3550, 2357, 2826, 2975]
        if input_M is None:
            self.ant_num = 15
        if input_Alpha is None:
            self.Alpha = 1
        if input_Beta is None:
            self.Beta = 2
        if input_Pho is None:
            self.Pho = 0.5
        self.city_num = len(self.X)
        self.city_index = np.arange(self.city_num)
        self.distance = np.zeros((self.city_num, self.city_num))
        for i in range(self.city_num):
            self.distance[i, i] = float('inf')
            for j in range(i + 1, self.city_num):
                dis = calculate_distance(self.X[i], self.Y[i], self.X[j], self.Y[j])
                self.distance[i, j] = dis
                self.distance[j, i] = dis
        (direction, _) = np.unravel_index(self.distance.argmin(), self.distance.shape)
        # path = [direction]
        total_distance = 0
        temp_distance = self.distance.copy()
        for i in range(self.city_num - 1):
            min_val = min(temp_distance[direction, :])
            min_index = list(temp_distance[direction, :]).index(min_val)
            # path.append(min_index)
            total_distance += min_val
            temp_distance[:, direction] = float('inf')
            direction = min_index
        (direction, _) = np.unravel_index(self.distance.argmin(), self.distance.shape)
        # print(total_distance)
        total_distance += self.distance[min_index, direction]
        # print(total_distance)
        self.pheromones_map = np.ones((self.city_num, self.city_num)) * self.ant_num / direction
        self.turn = 50
        self.Q = 1

    def AntColonyAlgorithm(self):
        for _ in range(self.turn):
            ants_path = []
            for j in range(self.ant_num):
                ants_path.append([random.randint(0, self.city_num - 1)])
            ants_map = []
            for j in range(self.ant_num):
                ants_map.append(self.distance.copy())
                ants_map[j][:, ants_path[j][0]] = float('inf')
            ants_distance = [0] * self.ant_num
            for i in range(self.city_num - 1):
                for j in range(self.ant_num):
                    prob = (self.pheromones_map[ants_path[j][i], :] ** self.Alpha) * \
                           ((np.ones(self.city_num) / ants_map[j][ants_path[j][i], :]) ** self.Beta)
                    choice = random.choices(self.city_index, prob, k=1)[0]
                    ants_map[j][:, choice] = float('inf')
                    ants_path[j].append(choice)
                    ants_distance[j] += self.distance[ants_path[j][i], choice]
            for j in range(self.ant_num):
                ants_path[j].append(ants_path[j][0])
                ants_distance[j] += self.distance[ants_path[j][-2], ants_path[j][-1]]
            for i in range(self.city_num):
                for j in range(i + 1, self.city_num):
                    self.pheromones_map[i][j] *= 1 - self.Pho
                    for k in range(self.ant_num):
                        self.pheromones_map[i][j] += self.plus_pheromones(i, j, ants_path[k], ants_distance[k])
                    self.pheromones_map[j][i] = self.pheromones_map[i][j]
            print("Turn num:", _)
            print("Min of ants_distance: ", min(ants_distance))
            print("Min ant path: ", ants_path[ants_distance.index(min(ants_distance))])

    def plus_pheromones(self, x, y, ant_path, ant_distance):
        for i in range(self.city_num):
            if (ant_path[i] == x and ant_path[i + 1] == y) or (ant_path[i] == y and ant_path[i + 1] == x):
                return self.Q / ant_distance
        return 0
