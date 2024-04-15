import numpy as np
'''
simply softmax with no grad calculate
'''

def my_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    print(my_softmax([3, 3, 4, 0]))
