# AI Lab05 Report

---

[toc]

Author: Songqing Zhao, Minzu University of China 

Written at Nov 8^th^, 2021

https://github.com/LibertyChaser/AI-Labs

> [Evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm)
>
> [How to convert a binary (string) into a float value?](https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value)
>
> [random](https://docs.python.org/3/library/random.html#module-random) — Generate pseudo-random numbers

---

## Lab Purpose

## Lab Background 

### Evolutionary algorithm

In [computational intelligence](https://en.wikipedia.org/wiki/Computational_intelligence) (CI), an **evolutionary algorithm** (**EA**) is a [subset](https://en.wikipedia.org/wiki/Subset)of [evolutionary computation](https://en.wikipedia.org/wiki/Evolutionary_computation),[[1\]](https://en.wikipedia.org/wiki/Evolutionary_algorithm#cite_note-EVOALG-1) a generic population-based [metaheuristic](https://en.wikipedia.org/wiki/Metaheuristic)[optimization](https://en.wikipedia.org/wiki/Optimization_(mathematics)) [algorithm](https://en.wikipedia.org/wiki/Algorithm). An EA uses mechanisms inspired by [biological evolution](https://en.wikipedia.org/wiki/Biological_evolution), such as [reproduction](https://en.wikipedia.org/wiki/Reproduction), [mutation](https://en.wikipedia.org/wiki/Mutation), [recombination](https://en.wikipedia.org/wiki/Genetic_recombination), and [selection](https://en.wikipedia.org/wiki/Natural_selection). [Candidate solutions](https://en.wikipedia.org/wiki/Candidate_solution) to the [optimization problem](https://en.wikipedia.org/wiki/Optimization_problem) play the role of individuals in a population, and the [fitness function](https://en.wikipedia.org/wiki/Fitness_function) determines the quality of the solutions (see also [loss function](https://en.wikipedia.org/wiki/Loss_function)). [Evolution](https://en.wikipedia.org/wiki/Evolution) of the population then takes place after the repeated application of the above operators.

Evolutionary algorithms often perform well approximating solutions to all types of problems because they ideally do not make any assumption about the underlying [fitness landscape](https://en.wikipedia.org/wiki/Fitness_landscape). Techniques from evolutionary algorithms applied to the modeling of biological evolution are generally limited to explorations of [microevolutionary processes](https://en.wikipedia.org/wiki/Microevolution) and planning models based upon cellular processes. In most real applications of EAs, computational complexity is a prohibiting factor.[[2\]](https://en.wikipedia.org/wiki/Evolutionary_algorithm#cite_note-VLSI-2) In fact, this computational complexity is due to fitness function evaluation. [Fitness approximation](https://en.wikipedia.org/wiki/Fitness_approximation) is one of the solutions to overcome this difficulty. However, seemingly simple EA can solve often complex problems;[*[citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)*] therefore, there may be no direct link between algorithm complexity and problem complexity.

#### Types

Similar techniques differ in [genetic representation](https://en.wikipedia.org/wiki/Genetic_representation) and other implementation details, and the nature of the particular applied problem.

- [Genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) – This is the most popular type of EA. One seeks the solution of a problem in the form of strings of numbers (traditionally binary, although the best representations are usually those that reflect something about the problem being solved),[[2\]](https://en.wikipedia.org/wiki/Evolutionary_algorithm#cite_note-VLSI-2) by applying operators such as recombination and mutation (sometimes one, sometimes both). This type of EA is often used in [optimization](https://en.wikipedia.org/wiki/Optimization_(mathematics)) problems.
- [Genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) – Here the solutions are in the form of computer programs, and their fitness is determined by their ability to solve a computational problem. There are many variants of Genetic Programming, including [Cartesian genetic programming](https://en.wikipedia.org/wiki/Cartesian_genetic_programming), [Gene expression programming](https://en.wikipedia.org/wiki/Gene_expression_programming), [Grammatical Evolution](https://en.wikipedia.org/wiki/Grammatical_Evolution), [Linear genetic programming](https://en.wikipedia.org/wiki/Linear_genetic_programming), [Multi expression programming](https://en.wikipedia.org/wiki/Multi_expression_programming) etc.
- [Evolutionary programming](https://en.wikipedia.org/wiki/Evolutionary_programming) – Similar to genetic programming, but the structure of the program is fixed and its numerical parameters are allowed to evolve.
- [Evolution strategy](https://en.wikipedia.org/wiki/Evolution_strategy) – Works with vectors of real numbers as representations of solutions, and typically uses self-adaptive mutation rates.
- [Differential evolution](https://en.wikipedia.org/wiki/Differential_evolution) – Based on vector differences and is therefore primarily suited for [numerical optimization](https://en.wikipedia.org/wiki/Numerical_optimization) problems.
- [Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution) – Similar to genetic programming but the genomes represent artificial neural networks by describing structure and connection weights. The genome encoding can be direct or indirect.
- [Learning classifier system](https://en.wikipedia.org/wiki/Learning_classifier_system) – Here the solution is a set of classifiers (rules or conditions). A Michigan-LCS evolves at the level of individual classifiers whereas a Pittsburgh-LCS uses populations of classifier-sets. Initially, classifiers were only binary, but now include real, neural net, or [S-expression](https://en.wikipedia.org/wiki/S-expression) types. Fitness is typically determined with either a strength or accuracy based [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) or [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) approach.

### [Genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)

In [computer science](https://en.wikipedia.org/wiki/Computer_science) and [operations research](https://en.wikipedia.org/wiki/Operations_research), a **genetic algorithm** (**GA**) is a [metaheuristic](https://en.wikipedia.org/wiki/Metaheuristic) inspired by the process of [natural selection](https://en.wikipedia.org/wiki/Natural_selection) that belongs to the larger class of [evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm) (EA). Genetic algorithms are commonly used to generate high-quality solutions to [optimization](https://en.wikipedia.org/wiki/Optimization_(mathematics)) and [search problems](https://en.wikipedia.org/wiki/Search_algorithm) by relying on biologically inspired operators such as [mutation](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)), [crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) and [selection](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)).[[1\]](https://en.wikipedia.org/wiki/Genetic_algorithm#cite_note-FOOTNOTEMitchell19962-1) Some examples of GA applications include optimizing [decision trees](https://en.wikipedia.org/wiki/Decision_tree_learning) for better performance, automatically solve [sudoku puzzles](https://en.wikipedia.org/wiki/Sudoku_solving_algorithms),[[2\]](https://en.wikipedia.org/wiki/Genetic_algorithm#cite_note-2)[hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization), etc.

In a genetic algorithm, a [population](https://en.wikipedia.org/wiki/Population) of [candidate solutions](https://en.wikipedia.org/wiki/Candidate_solution) (called individuals, creatures, or [phenotypes](https://en.wikipedia.org/wiki/Phenotype)) to an optimization problem is evolved toward better solutions. Each candidate solution has a set of properties (its [chromosomes](https://en.wikipedia.org/wiki/Chromosome) or [genotype](https://en.wikipedia.org/wiki/Genotype)) which can be mutated and altered; traditionally, solutions are represented in binary as strings of 0s and 1s, but other encodings are also possible.[[3\]](https://en.wikipedia.org/wiki/Genetic_algorithm#cite_note-FOOTNOTEWhitley199466-3)

The evolution usually starts from a population of randomly generated individuals, and is an [iterative process](https://en.wikipedia.org/wiki/Iteration), with the population in each iteration called a *generation*. In each generation, the [fitness](https://en.wikipedia.org/wiki/Fitness_(biology)) of every individual in the population is evaluated; the fitness is usually the value of the [objective function](https://en.wikipedia.org/wiki/Objective_function) in the optimization problem being solved. The more fit individuals are [stochastically](https://en.wikipedia.org/wiki/Stochastics) selected from the current population, and each individual's genome is modified ([recombined](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm))and possibly randomly mutated) to form a new generation. The new generation of candidate solutions is then used in the next iteration of the [algorithm](https://en.wikipedia.org/wiki/Algorithm). Commonly, the algorithm terminates when either a maximum number of generations has been produced, or a satisfactory fitness level has been reached for the population.

A typical genetic algorithm requires:

1. a [genetic representation](https://en.wikipedia.org/wiki/Genetic_representation) of the solution domain,
2. a [fitness function](https://en.wikipedia.org/wiki/Fitness_function) to evaluate the solution domain.

A standard representation of each candidate solution is as an [array of bits](https://en.wikipedia.org/wiki/Bit_array) (also called *bit set* or *bit string*).[[3\]](https://en.wikipedia.org/wiki/Genetic_algorithm#cite_note-FOOTNOTEWhitley199466-3) Arrays of other types and structures can be used in essentially the same way. The main property that makes these genetic representations convenient is that their parts are easily aligned due to their fixed size, which facilitates simple [crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) operations. Variable length representations may also be used, but crossover implementation is more complex in this case. Tree-like representations are explored in [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) and graph-form representations are explored in [evolutionary programming](https://en.wikipedia.org/wiki/Evolutionary_programming); a mix of both linear chromosomes and trees is explored in [gene expression programming](https://en.wikipedia.org/wiki/Gene_expression_programming).

Once the genetic representation and the fitness function are defined, a GA proceeds to initialize a population of solutions and then to improve it through repetitive application of the mutation, crossover, inversion and selection operators.

## Lab Procedure

### Basic Info

```python
# -*- coding: utf-8 -*-
"""
Created on Nov 8 2021
Finished on Nov 9, 2021
@author: Songqing Zhao, Minzu University of China
"""
import random
import struct
from codecs import decode
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean

```

### Initialization

The population size depends on the nature of the problem, but typically contains several hundreds or thousands of possible solutions. Often, the initial population is generated randomly, allowing the entire range of possible solutions (the [search space](https://en.wikipedia.org/wiki/Feasible_region)). Occasionally, the solutions may be "seeded" in areas where optimal solutions are likely to be found.


```python
def float_to_bin(value):
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def fitness_function(x_1, x_2):
    return 1 / (x_1 ** 2 + x_2 ** 2 + 1)


class GeneticAlgorithm:
    def __init__(self, upper=5, lower=-5, population_size=20, precision=4, generation=40):
        self.UPPER = upper
        self.LOWER = lower
        self.POPULATION = population_size
        self.PRECISION = precision
        self.FLOATING_POINT = 64
        self.SINGLE_POINT_CROSSOVER_NUM = self.FLOATING_POINT
        self.SINGLE_POINT_CROSSOVER_PROB = 0.7
        self.SINGLE_MUTATION_PROB = 0.1
        self.GENERATION_SIZE = generation
        self.population_1 = [round(random.random() * (self.UPPER - self.LOWER) + self.LOWER, self.PRECISION) for _ in
                             range(self.POPULATION)]
        self.population_2 = [round(random.random() * (self.UPPER - self.LOWER) + self.LOWER, self.PRECISION) for _ in
                             range(self.POPULATION)]
        self.first_coding = self.coding(self.population_1, self.population_2)

    def coding(self, input_1, input_2):
        return [float_to_bin(input_1[i]) + float_to_bin(input_2[i]) for i in range(self.POPULATION)]
```

### Selection

During each successive generation, a portion of the existing population is [selected](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)) to breed a new generation. Individual solutions are selected through a *fitness-based* process, where [fitter](https://en.wikipedia.org/wiki/Fitness_(biology)) solutions (as measured by a [fitness function](https://en.wikipedia.org/wiki/Fitness_function)) are typically more likely to be selected. Certain selection methods rate the fitness of each solution and preferentially select the best solutions. Other methods rate only a random sample of the population, as the former process may be very time-consuming.

The fitness function is defined over the genetic representation and measures the *quality* of the represented solution. The fitness function is always problem dependent. For instance, in the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) one wants to maximize the total value of objects that can be put in a knapsack of some fixed capacity. A representation of a solution might be an array of bits, where each bit represents a different object, and the value of the bit (0 or 1) represents whether or not the object is in the knapsack. Not every such representation is valid, as the size of objects may exceed the capacity of the knapsack. The *fitness* of the solution is the sum of values of all objects in the knapsack if the representation is valid, or 0 otherwise.

In some problems, it is hard or even impossible to define the fitness expression; in these cases, a [simulation](https://en.wikipedia.org/wiki/Computer_simulation) may be used to determine the fitness function value of a [phenotype](https://en.wikipedia.org/wiki/Phenotype) (e.g. [computational fluid dynamics](https://en.wikipedia.org/wiki/Computational_fluid_dynamics) is used to determine the air resistance of a vehicle whose shape is encoded as the phenotype), or even [interactive genetic algorithms](https://en.wikipedia.org/wiki/Interactive_evolutionary_computation) are used.

```python
    def make_evaluation(self, code):
        l1 = [bin_to_float(i[0: self.FLOATING_POINT]) for i in code]
        l2 = [bin_to_float(i[self.FLOATING_POINT:]) for i in code]
        return {code[i]: fitness_function(l1[i], l2[i]) for i in range(self.POPULATION)}

    def survival(self, fitness):
        return random.choices(list(fitness.keys()), list(fitness.values()), k=self.POPULATION)

    def decode_first(self, code):
        return bin_to_float(code[0:self.FLOATING_POINT])

    def decode_last(self, code):
        return bin_to_float(code[self.FLOATING_POINT:])

    def check_in_range(self, input_element):
        if self.LOWER <= input_element <= self.UPPER:
            return 1
        return 0
```

### Genetic operators

Main articles: [Crossover (genetic algorithm)](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) and [Mutation (genetic algorithm)](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm))

The next step is to generate a second generation population of solutions from those selected through a combination of [genetic operators](https://en.wikipedia.org/wiki/Genetic_operator): [crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) (also called recombination), and [mutation](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)).

For each new solution to be produced, a pair of "parent" solutions is selected for breeding from the pool selected previously. By producing a "child" solution using the above methods of crossover and mutation, a new solution is created which typically shares many of the characteristics of its "parents". New parents are selected for each new child, and the process continues until a new population of solutions of appropriate size is generated. Although reproduction methods that are based on the use of two parents are more "biology inspired", some research[[4\]](https://en.wikipedia.org/wiki/Genetic_algorithm#cite_note-4)[[5\]](https://en.wikipedia.org/wiki/Genetic_algorithm#cite_note-5) suggests that more than two "parents" generate higher quality chromosomes.

These processes ultimately result in the next generation population of chromosomes that is different from the initial generation. Generally, the average fitness will have increased by this procedure for the population, since only the best organisms from the first generation are selected for breeding, along with a small proportion of less fit solutions. These less fit solutions ensure genetic diversity within the genetic pool of the parents and therefore ensure the genetic diversity of the subsequent generation of children.

Opinion is divided over the importance of crossover versus mutation. There are many references in [Fogel](https://en.wikipedia.org/wiki/David_B._Fogel) (2006) that support the importance of mutation-based search.

Although crossover and mutation are known as the main genetic operators, it is possible to use other operators such as regrouping, colonization-extinction, or migration in genetic algorithms.[*[citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)*]

It is worth tuning parameters such as the [mutation](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)) probability, [crossover](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)) probability and population size to find reasonable settings for the problem class being worked on. A very small mutation rate may lead to [genetic drift](https://en.wikipedia.org/wiki/Genetic_drift) (which is non-[ergodic](https://en.wikipedia.org/wiki/Ergodicity) in nature). A recombination rate that is too high may lead to premature convergence of the genetic algorithm. A mutation rate that is too high may lead to loss of good solutions, unless [elitist selection](https://en.wikipedia.org/wiki/Genetic_algorithm#Elitism) is employed. An adequate population size ensures sufficient genetic diversity for the problem at hand, but can lead to a waste of computational resources if set to a value larger than required.

```python
    def single_point_crossover(self, input_survivors):
        survivors = input_survivors
        for _ in range(self.SINGLE_POINT_CROSSOVER_NUM):
            i = int(random.random() * self.POPULATION - 1)
            if random.random() < self.SINGLE_POINT_CROSSOVER_PROB:
                change_bit = int(random.random() * (self.FLOATING_POINT * 2) - 1)
                if change_bit == 0:
                    temp1 = survivors[i + 1][change_bit] + survivors[i][1:]
                    temp2 = survivors[i][change_bit] + survivors[i + 1][1:]
                elif change_bit == self.FLOATING_POINT * 2 - 1:
                    temp1 = survivors[i][0:change_bit] + survivors[i + 1][change_bit]
                    temp2 = survivors[i + 1][0:change_bit] + survivors[i][change_bit]
                else:
                    temp1 = survivors[i][0:change_bit] + survivors[i + 1][change_bit] + survivors[i][change_bit + 1:]
                    temp2 = survivors[i + 1][0:change_bit] + survivors[i][change_bit] + survivors[i + 1][change_bit + 1:]
                if self.checkout(temp1) and self.checkout(temp2):
                    survivors[i] = temp1
                    survivors[i + 1] = temp2
                else:
                    _ -= 1
        return survivors

    def single_point_mutation(self, input_survivors):
        survivors = input_survivors
        for i in range(self.POPULATION):
            if random.random() < self.SINGLE_MUTATION_PROB:
                change_bit = int(random.random() * (self.FLOATING_POINT * 2) - 1)
                l = ['1', '0']
                temp_bit = l[int(survivors[i][change_bit])]
                if change_bit == 0:
                    temp = temp_bit + survivors[i][1:]
                elif change_bit == self.FLOATING_POINT * 2 - 1:
                    temp = survivors[i][0:change_bit] + temp_bit
                else:
                    temp = survivors[i][0:change_bit] + temp_bit + survivors[i][change_bit + 1:]
                if self.checkout(temp):
                    survivors[i] = temp
                else:
                    i -= 1
        return survivors

```

### Termination

This generational process is repeated until a termination condition has been reached. Common terminating conditions are:

- A solution is found that satisfies minimum criteria
- Fixed number of generations reached
- Allocated budget (computation time/money) reached
- The highest ranking solution's fitness is reaching or has reached a plateau such that successive iterations no longer produce better results
- Manual inspection
- Combinations of the above

```python
  	def GA(self):
        fitness = [self.make_evaluation(self.first_coding)]
        for i in range(self.GENERATION_SIZE):
            print("The ", i + 1, " Generation:")
            survivors = self.survival(fitness[i])
            crossover = self.single_point_crossover(survivors)
            mutation = self.single_point_mutation(crossover)
            fitness.append(self.make_evaluation(mutation))
            print("Current generation max evaluation: ", max(fitness[i+1].values()), "\n")

        generation = np.arange(1, self.GENERATION_SIZE + 1, 1)
        max_eval = [max(fitness[i].values()) for i in range(self.GENERATION_SIZE)]
        mean_eval = [sum(fitness[i].values()) / len(fitness[i].values()) for i in range(self.GENERATION_SIZE)]
        min_eval = [min(fitness[i].values()) for i in range(self.GENERATION_SIZE)]
```
### DIsplay the result
```python
        plt.plot(generation, max_eval, 'b', generation, mean_eval, 'r', generation, min_eval, 'g')
        plt.legend(['Max Fitness', 'Mean Fitness', 'Min Fitness'])
        plt.xlabel("Generation Num")
        plt.ylabel("Fitness")
        plt.title("Fitness with Generation")
        plt.text(self.GENERATION_SIZE * 0.7, 0.3, "$y = \\frac{1}{x_1 ^ 2 + x_2 ^ 2 + 1}$", fontsize=15)
        plt.grid(True)
        plt.show()
        plt.savefig('fitness_with_generation.svg')

        print("Last Generation's fittest child are: ")
        a = max(fitness[self.GENERATION_SIZE], key=fitness[self.GENERATION_SIZE].get)
        print("x_1 = ", round(self.decode_last(a), self.PRECISION), "\nx_2 = ", round(self.decode_first(a), self.PRECISION))
        
```

## Lab Result

### Test code

`main.py`

```python

import random

import matplotlib.pyplot as plt

from GeneticAlgorithm import *
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    a = GeneticAlgorithm()
    GeneticAlgorithm.GA(a)

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    x_1 = np.arange(a.LOWER, a.UPPER, 0.1)
    x_2 = np.arange(a.LOWER, a.UPPER, 0.1)
    x_1, x_2 = np.meshgrid(x_1, x_2)
    ax.plot_surface(x_1, x_2, fitness_function(x_1, x_2), rstride=1, cstride=1)
    ax.view_init(elev=30, azim=125)
    plt.show()
```

### Result Visualization

#### Fitness plot

<img src="AI Lab05 Report.assets/fitness_plot.png" alt="fitness_plot" style="zoom:50%;" />

#### Fitness wiith generation

<img src="AI Lab05 Report.assets/fitness_with_generation.png" alt="fitness_with_generation" style="zoom:50%;" />

### Output

One of the output. Not correspond to the image.

```python
The  1  Generation:
Current generation max evaluation:  0.7283851170234454 

The  2  Generation:
Current generation max evaluation:  0.8865956603312688 

The  3  Generation:
Current generation max evaluation:  0.8876004856793579 

The  4  Generation:
Current generation max evaluation:  0.8876004856793579 

The  5  Generation:
Current generation max evaluation:  0.8876004856793579 

The  6  Generation:
Current generation max evaluation:  0.88760060811082 

The  7  Generation:
Current generation max evaluation:  0.8876009754048972 

The  8  Generation:
Current generation max evaluation:  0.9551341012614456 

The  9  Generation:
Current generation max evaluation:  0.9551341012614455 

The  10  Generation:
Current generation max evaluation:  0.9551341012614382 

The  11  Generation:
Current generation max evaluation:  0.9527615620380182 

The  12  Generation:
Current generation max evaluation:  0.9527615620380182 

The  13  Generation:
Current generation max evaluation:  0.9527615620380182 

The  14  Generation:
Current generation max evaluation:  0.8876148379850519 

The  15  Generation:
Current generation max evaluation:  0.8876009753524988 

The  16  Generation:
Current generation max evaluation:  0.9603318890072965 

The  17  Generation:
Current generation max evaluation:  0.9603319113505475 

The  18  Generation:
Current generation max evaluation:  0.9603318890066147 

The  19  Generation:
Current generation max evaluation:  0.9799866795326029 

The  20  Generation:
Current generation max evaluation:  0.979986679514371 

The  21  Generation:
Current generation max evaluation:  0.979986679514371 

The  22  Generation:
Current generation max evaluation:  0.97998667951665 

The  23  Generation:
Current generation max evaluation:  0.9799866795143708 

The  24  Generation:
Current generation max evaluation:  0.9805193888546964 

The  25  Generation:
Current generation max evaluation:  0.9805193888546964 

The  26  Generation:
Current generation max evaluation:  0.9805193888546964 

The  27  Generation:
Current generation max evaluation:  0.9799531611365958 

The  28  Generation:
Current generation max evaluation:  0.9145145150928657 

The  29  Generation:
Current generation max evaluation:  0.8944717583515865 

The  30  Generation:
Current generation max evaluation:  0.8944717583515865 

The  31  Generation:
Current generation max evaluation:  0.8944717583515865 

The  32  Generation:
Current generation max evaluation:  0.8944874826794799 

The  33  Generation:
Current generation max evaluation:  0.8944717583515865 

The  34  Generation:
Current generation max evaluation:  0.8944707059336631 

The  35  Generation:
Current generation max evaluation:  0.9029437797226654 

The  36  Generation:
Current generation max evaluation:  0.9029437797226654 

The  37  Generation:
Current generation max evaluation:  1.0 

The  38  Generation:
Current generation max evaluation:  1.0 

The  39  Generation:
Current generation max evaluation:  1.0 

The  40  Generation:
Current generation max evaluation:  1.0 

Last Generation's fittest child are: 
x_1 =  0.0 
x_2 =  -0.0
```

