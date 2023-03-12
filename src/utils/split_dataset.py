from .data_process import split_dataset
from .config import gener_config as config


if __name__ == '__main__':
    print('Split rate: ', config['split_rate'])
    split_dataset(config)