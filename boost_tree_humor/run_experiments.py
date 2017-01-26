import os
import sys
import logging
import itertools

import numpy as np

from tree_model import ex


def main():
    nb_samples = 2000

    default_params = {
        'silent': 1,
    }

    params = {
        'max_depth': lambda: int(np.random.uniform(2, 50)),
        'eta': lambda: 10 ** np.random.uniform(-6, -1.5),
        'gamma': lambda: int(np.random.uniform(1, 20)),
        'reg_lambda': lambda: 10 ** np.random.uniform(-6, -1.5),
        'num_round': lambda: int(np.random.uniform(1, 20)),
    }

    for i in range(nb_samples):
        logging.info('Starting sample %s/%s [%.2f]', i, nb_samples, i / nb_samples)

        current_params = default_params.copy()
        for param_name, param_sample in params.items():
            param_value = param_sample()
            current_params[param_name] = param_value

        ex.run_command(
            ex.default_command,
            config_updates=current_params,
            args={'--force': True}
        )


if __name__ == '__main__':
    main()
