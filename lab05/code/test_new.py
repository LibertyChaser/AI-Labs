import random
import struct
from codecs import decode


# https://stackoverflow.com/questions/8751653/how-to-convert-a-binary-string-into-a-float-value/8762541
# https://docs.python.org/3/library/random.html

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


def evaluate_function(x_1, x_2):
    return 1 / (x_1 ** 2 + x_2 ** 2 + 1)


class GeneticAlgorithm:
    def __init__(self, upper=5, lower=-5, population_size=20, precision=4):
        self.UPPER = upper
        self.LOWER = lower
        self.POPULATION = population_size
        self.PRECISION = precision
        self.FLOATING_POINT = 64
        self.single_point_crossover_num = self.FLOATING_POINT
        self.single_point_crossover_prob = 0.7
        self.mutation_prob = 0.1
        self.population_1 = [round(random.random() * (self.UPPER - self.LOWER) + self.LOWER, precision) for _ in
                             range(self.POPULATION)]
        self.population_2 = [round(random.random() * (self.UPPER - self.LOWER) + self.LOWER, precision) for _ in
                             range(self.POPULATION)]
        self.first_coding = self.coding(self.population_1, self.population_2)

    def coding(self, input_1, input_2):
        return [float_to_bin(input_1[i]) + float_to_bin(input_2[i]) for i in range(self.POPULATION)]

    def decoding_and_evaluate(self, code):
        l1 = [bin_to_float(i[0: self.FLOATING_POINT]) for i in code]
        l2 = [bin_to_float(i[self.FLOATING_POINT:]) for i in code]
        return {code[i]: evaluate_function(l1[i], l2[i]) for i in range(self.POPULATION)}

    def survival(self, fitness):
        return random.choices(list(fitness.keys()), list(fitness.values()), k=self.POPULATION)

    def single_point_crossover(self, input_survivors):
        survivors = input_survivors
        for _ in range(self.single_point_crossover_num):
            i = int(random.random() * self.POPULATION - 1)
            if random.random() < self.single_point_crossover_prob:
                change_bit = int(random.random() * (self.FLOATING_POINT * 2) - 1)
                temp = survivors[i][change_bit]
                if change_bit == 0:
                    survivors[i] = survivors[i+1][change_bit] + survivors[i][1:]
                    survivors[i+1] = temp + survivors[i+1][1:]
                elif change_bit == self.FLOATING_POINT * 2 - 1:
                    survivors[i] = survivors[i][0:change_bit] + survivors[i+1][change_bit]
                    survivors[i+1] = survivors[i+1][0:change_bit] + temp
                else:
                    survivors[i] = survivors[i][0:change_bit] + survivors[i + 1][change_bit] + survivors[i][change_bit+1:]
                    survivors[i+1] = survivors[i+1][0:change_bit] + temp + survivors[i+1][change_bit+1:]
        return survivors

    def single_point_mutation(self, input_survivors):
        survivors = input_survivors
        
        return survivors

    def test_gen(self):
        first_evaluate = self.decoding_and_evaluate(self.first_coding)
        first_survivors = self.survival(first_evaluate)
        print(first_survivors)
        print(self.single_point_crossover(first_survivors))
        pass

a = GeneticAlgorithm()
GeneticAlgorithm.test_gen(a)
