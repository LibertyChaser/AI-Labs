import random

if __name__ == '__main__':
    for i in range(3):
        print(i)
        while 1:
            if random.random() <= 0.4:
                break
            print(22)
    pass
