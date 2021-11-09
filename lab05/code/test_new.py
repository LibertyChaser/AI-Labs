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
        self.SINGLE_POINT_CROSSOVER_NUM = self.FLOATING_POINT
        self.SINGLE_POINT_CROSSOVER_PROB = 0.7
        self.SINGLE_MUTATION_PROB = 0.9
        self.population_1 = [round(random.random() * (self.UPPER - self.LOWER) + self.LOWER, self.PRECISION) for _ in
                             range(self.POPULATION)]
        self.population_2 = [round(random.random() * (self.UPPER - self.LOWER) + self.LOWER, self.PRECISION) for _ in
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

    def decode_first(self, code):
        return bin_to_float(code[0:self.FLOATING_POINT])

    def decode_last(self, code):
        return bin_to_float(code[self.FLOATING_POINT:])

    def check_in_range(self, input_element):
        if self.LOWER <= input_element <= self.UPPER:
            return 1
        return 0

    def checkout(self, code):
        if self.check_in_range(self.decode_first(code)) and self.check_in_range(self.decode_last(code)):
            return 1
        return 0

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

    def test_gen(self):
        first_evaluate = self.decoding_and_evaluate(self.first_coding)

        first_survivors = self.survival(first_evaluate)
        print(first_survivors)
        print(self.decode_first(first_survivors[2]), self.decode_last(first_survivors[2]))
        print('----')
        crossover = self.single_point_crossover(first_survivors)
        print(crossover)
        print(self.decode_first(crossover[2]), self.decode_last(crossover[2]))
        print('----')
        mutation = self.single_point_mutation(crossover)
        print(mutation)
        print(self.decode_first(mutation[2]), self.decode_last(mutation[2]))
        pass


a = GeneticAlgorithm()
GeneticAlgorithm.test_gen(a)
