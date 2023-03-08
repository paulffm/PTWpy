#from scipy.stats import uniform, randint
from random import randint, uniform, choice
from sklearn.utils.fixes import loguniform
import numpy as np
import math
import random





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

    param_distribution = {'max_depth': list(range(6, 15)),
              'learning_rate': [0.050, 0.055, 0.0575, 0.06, 0.065, 0.07, 0.075, 0.8],
              'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, ],
              'min_child_weight': list(range(1, 2))
              }
    print(param_distribution)
    param = {k: random.choice(dist) for k, dist in param_distribution.items()}
    print(param)

    param_distributions = {
        'learning_rate': uniform(0.001, 0.1),
        'num_layers': randint(1, 5),
        'num_neurons': randint(10, 100)
    }
    param1 = {k: dist.rvs() for k, dist in param_distributions.items()}
    print(param1)

    param_distribution = {'max_depth': randint(6, 15),
              'learning_rate': [0.050, 0.055, 0.0575, 0.06, 0.065, 0.07, 0.075, 0.8],
              'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, ],
              'min_child_weight': randint(1, 2)
              }
    params = {k: dist.rvs() for k, dist in param_distribution.items()}
    print(params)



if __name__ == '__main__':
    main()