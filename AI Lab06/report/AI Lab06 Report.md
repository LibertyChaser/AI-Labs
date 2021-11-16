# AI Lab06 Report

---

[toc]

Author: Songqing Zhao, Minzu University of China 

Written at Nov 16^th^, 2021

https://github.com/LibertyChaser/AI-Labs

>[Ant colony optimization algorithms](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)

---

## Lab Purpose

Solving TSP problem with ant colony algorithm

## Lab Principle

### Introduction

In [computer science](https://en.wikipedia.org/wiki/Computer_science) and [operations research](https://en.wikipedia.org/wiki/Operations_research), the **ant colony optimization**[algorithm](https://en.wikipedia.org/wiki/Algorithm) (**ACO**) is a [probabilistic](https://en.wikipedia.org/wiki/Probability) technique for solving computational problems which can be reduced to finding good paths through [graphs](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)). Artificial ants stand for [multi-agent](https://en.wikipedia.org/wiki/Multi-agent) methods inspired by the behavior of real ants. The pheromone-based communication of biological [ants](https://en.wikipedia.org/wiki/Ant) is often the predominant paradigm used.[[2\]](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#cite_note-2) Combinations of artificial ants and [local search](https://en.wikipedia.org/wiki/Local_search_(optimization)) algorithms have become a method of choice for numerous optimization tasks involving some sort of [graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)), e.g., [vehicle routing](https://en.wikipedia.org/wiki/Vehicle_routing_problem) and internet [routing](https://en.wikipedia.org/wiki/Routing).

### Parameters

1. Pheromone Inspiration Factor Alpha

   -  Reflects the intensity of the role of random factors in the path search of the ant colony;

   -  The larger the value of α, the more likely the ants will choose the path they have walked before, and the randomness of the search will be weakened; 

   -  When α is too large, the search of the ant colony will fall into the local optimum prematurely.
2. Expected value heuristic factor β
    - Reflects the strength of the role of a priori and deterministic factors in the path search of the ant colony;
     - The larger the value of β is, the more likely the ant will choose the local shortest path at a certain local point;
     - Although the convergence speed of the search can be accelerated, it weakens the randomness of the ant colony in the search process of the optimal path, and it is easy to fall into the local optimum.
3. Pheromone volatility 1-ρ
    - When the scale of the problem to be dealt with is relatively large, the amount of information on the path (feasible solution) that has never been searched will be reduced to close to 0, thus reducing the global search ability of the algorithm;
    - When 1-ρ is too large, the possibility that the previously searched path will be selected again is too large, which will also affect the random performance and global search ability of the algorithm;
    - Conversely, by reducing the pheromone volatility 1-ρ, although the random performance and global search ability of the algorithm can be improved, it will also reduce the convergence speed of the algorithm.

## Lab Procedure

```python
import random

import matplotlib.pyplot as plt
import numpy as np

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  
```

### Initialization

#### Basic perimeters

```python
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
        self.turn = 80
        self.Q = 1
````
#### Distance matrix
```python
        self.distance = np.zeros((self.city_num, self.city_num))
        for i in range(self.city_num):
            self.distance[i, i] = float('inf')
            for j in range(i + 1, self.city_num):
                dis = calculate_distance(self.X[i], self.Y[i], self.X[j], self.Y[j])
                self.distance[i, j] = dis
                self.distance[j, i] = dis
```
#### Get pheromones map
```python
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
```

### Select first departure cities

```python
    def AntColonyAlgorithm(self):
        min_dis = []
        mean_dis = []
        max_dis = []
        min_path = []
        for _ in range(self.turn):
            ants_path = []
            for j in range(self.ant_num):
                ants_path.append([random.randint(0, self.city_num - 1)])
```

### Calculate the probability of ants choosing a city

```python
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
```
### Print result
```python
            print("Turn num:", _ + 1)
            print("Min of ant distance: ", min(ants_distance))
            print("Min ant path: ", ants_path[ants_distance.index(min(ants_distance))])
            min_dis.append(min(ants_distance))
            mean_dis.append(sum(ants_distance) / len(ants_distance))
            max_dis.append(max(ants_distance))
            min_path.append(ants_path[ants_distance.index(min(ants_distance))])
```

### Visualization

```python
        turns = np.arange(1, self.turn + 1)
        plt.plot(turns, max_dis, 'b', turns, mean_dis, 'r', turns, min_dis, 'g')
        plt.legend(['Max Distance', 'Mean Distance', 'Min Distance'])
        plt.xlabel("Turns")
        plt.ylabel("Distance")
        plt.title("Distance with Turn")
        # plt.text(self.GENERATION_SIZE * 0.7, 0.3, "$y = \\frac{1}{x_1 ^ 2 + x_2 ^ 2 + 1}$", fontsize=15)
        plt.grid(True)
        plt.show()

        print("\nMin distance :", min(min_dis))
        print("Min ant path: ", min_path[min_dis.index(min(min_dis))])
```

## Lab Result

### Test code

In `main.py`

```python
from TSP import *

if __name__ == '__main__':
    a = TSP().AntColonyAlgorithm()

```

### Python console result

Run `main.py`

```python
Turn num: 1
Min of ant distance:  22807.785257143812
Min ant path:  [3, 1, 22, 5, 4, 11, 13, 17, 2, 18, 16, 19, 23, 7, 8, 9, 15, 12, 6, 10, 28, 29, 26, 27, 24, 25, 21, 20, 30, 0, 14, 3]
Turn num: 2
Min of ant distance:  22304.32819341383
Min ant path:  [12, 1, 3, 15, 5, 4, 10, 6, 9, 8, 7, 18, 16, 26, 28, 29, 20, 21, 25, 27, 24, 19, 23, 2, 17, 22, 11, 13, 14, 0, 30, 12]
...
Turn num: 79
Min of ant distance:  17230.96694724903
Min ant path:  [23, 24, 19, 20, 21, 17, 2, 16, 18, 22, 10, 12, 11, 13, 14, 0, 28, 30, 29, 26, 27, 25, 15, 3, 1, 4, 5, 6, 7, 8, 9, 23]
Turn num: 80
Min of ant distance:  16748.65896812881
Min ant path:  [24, 23, 10, 22, 18, 16, 2, 17, 21, 20, 19, 25, 27, 26, 29, 28, 30, 0, 14, 13, 11, 12, 6, 5, 4, 15, 3, 1, 7, 8, 9, 24]

Min distance : 15881.14585443691
Min ant path:  [14, 13, 11, 12, 10, 22, 15, 3, 1, 6, 5, 4, 9, 8, 7, 18, 16, 2, 17, 21, 20, 19, 23, 24, 25, 27, 26, 29, 30, 28, 0, 14]
```

### Result Visualization

<img src="AI Lab06 Report.assets/distance_with_turn.png" alt="distance_with_turn" style="zoom:50%;" />

## Improvement and innovation

:)
