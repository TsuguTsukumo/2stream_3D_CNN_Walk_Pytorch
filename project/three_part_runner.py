"""
help function for experiment.

"""

import time, os
import subprocess
from argparse import ArgumentParser

VIDEO_LENGTH = ['1']
VIDEO_FRAME = ['8']
PART = ['A', 'B', 'C']

MAIN_FILE_PATH = '/workspace/Walk_Video_PyTorch/project/main.py'

def get_parameters():
    '''
    The parameters for the model training, can be called out via the --h menu
    '''
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'csn', 'r2plus1d', 'x3d', 'slowfast', 'c2d', 'i3d'])
    parser.add_argument('--model_depth', type=int, default=50, choices=[50, 101, 152], help='the depth of used model')

    # Training setting
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')

    # Transfor_learning
    parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')

    # pre process flag
    parser.add_argument('--pre_process_flag', action='store_true', help='if use the pre process video, which detection.')

    # Path
    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    return parser.parse_known_args()


if __name__ == '__main__':

    config, unkonwn = get_parameters()

    transfor_learning = config.transfor_learning
    pre_process_flag = config.pre_process_flag
    model = config.model

    symbol = '_'

    for part in PART:
        data_path = os.path.join('/workspace/data/three_part_split_pad_dataset', part)
        for length in VIDEO_LENGTH:
            for frames in VIDEO_FRAME:

                data = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
                
                version = symbol.join([data, part, length, frames])
                log_path = '/workspace/Walk_Video_PyTorch/logs/' + symbol.join([version, model]) + '.log'

                with open(log_path, 'w') as f:

                    # start one train.
                    subprocess.run(['python', MAIN_FILE_PATH,
                                    '--version', version,
                                    '--model', model,
                                    '--clip_duration', length,
                                    '--uniform_temporal_subsample_num', frames,
                                    '--gpu_num', str(config.gpu_num),
                                    '--pre_process_flag',
                                    '--transfor_learning',
                                    '--split_pad_data_path', data_path,
                                    ], stdout=f, stderr=f)

                print('*' * 100)
                print('finish part ', part)
                print('finish %s' % log_path)
