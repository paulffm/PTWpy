#from scipy.stats import uniform, randint
from random import randint, uniform, choice
from sklearn.utils.fixes import loguniform
import numpy as np
import math

class pattern_finder:
    def __init__(self, idx1, idx2):
        self.idx1 = idx1
        self.idx2 = idx2

    def find_motifs(self, data):
        self.pattern = data[self.idx1:self.idx2]




def main():
    '''params = {
        'unit1': 0,
        'unit2': 0,
        'unit3': 0,
        'activation': '',
        'learning_rate': 0,
        'layers1': 0,
        'layers2': 0,
        'nb_epoch': 0,
        'batch_size': 0,
        'kernel_initializer': '',
        # 'normalization': uniform(0, 1),
        # 'optimizerL': ['Adam', 'SGD', 'Adagrad', 'Adamax', 'Adadelta',
        # 'optimizerL': ['RMSprop', 'Nadam', 'Ftrl']}
    }
    lr = np.linspace(0.001, 0.1, num=50)
    print(lr)
    print(choice(['he_uniform', 'glorot_uniform']))
    print('lr', choice(lr))
    print(randint(50, 250))
    print(params)
    print(params['unit1'])
    params['unit1'] = 1
    print(params['unit1'])'''
    data = np.arange(0, 11)

    for i in range(5):
        X = data[i:i+2]
        print(f'{i+1}: {X}')

if __name__ == '__main__':
    main()