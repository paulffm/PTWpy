#from scipy.stats import uniform, randint
from random import randint, uniform, choice
from sklearn.utils.fixes import loguniform
import numpy as np
def loguniform(low=0, high=1):
    return np.exp(uniform(low, high))

def main():
    params = {
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
    print(params['unit1'])

if __name__ == '__main__':
    main()