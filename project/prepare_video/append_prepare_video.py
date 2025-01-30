"""
Append experiment.
Split whole dataset into three part, where split one patient into three part, from start to end.
split pad dataset path: /workspace/data/split_pad_dataset_512
three part split pad dataset path: /workspace/data/three_part_split_pad_dataset
"""

# %%
import os 
import sys 
import shutil

# there should exchange the path with yourself path.
sys.path.append('/workspace/Walk_Video_PyTorch/project')

import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
from torchvision.io import read_video, read_video_timestamps, write_video
import torch

from pytorchvideo.transforms.functional import uniform_temporal_subsample

# %%

def del_folder(path, *args):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    else:
        print("The target path already exists! where: %s " % os.path.join(path, *args))


# %%
def get_Diease_Path_List(data_path: str):
    '''
    get the prefix data path list, like "/workspace/data/dataset/train/ASD", len = 4

    Args:
        data_path (str): meta data path

    Returns:
        list: list of the prefix data path list, len=4
    '''

    diease_path_list = []

    for flag in ('train', 'val'):

        data_path_flag = os.path.join(data_path, flag)

        for diease_flag in (os.listdir(data_path_flag)):  # ASD, ASD_not

            data_path_diease_flag = os.path.join(data_path_flag, diease_flag)

            diease_path_list.append(data_path_diease_flag)

    return diease_path_list

# %%
def get_final_video_path_Dict(prefix_path_list: list):
    '''
    get the all final video full path, in dict. 
    the keys mean the unique data with the disease.
    the values mean in on data with disease, how many file they are, store in List.

    Args:
        prefix_path_list (list): the prefix path list, like /train/ASD; /train/ASD_not; /val/ASD; /val/ASD_not.

    Returns:
        Dict: the all final video full path, store in dict.
    '''

    final_video_path_Dict = {}


    compare_video_name_list = []

    # all video path
    video_path_list = os.listdir(prefix_path_list)
    video_path_list.sort()

    # start compare video path list
    for compare_video_name in video_path_list:

        compare_video_name_list.append(compare_video_name[:15])

    compare_video_file_name = sorted(list(set(compare_video_name_list)))

    for name in compare_video_file_name:

        video_same_path_list = []

        for video_path_name in video_path_list:

            now_video_path_name = video_path_name[:15]

            if now_video_path_name in name:

                # store the full path of unique data with diease in a list.
                video_same_path_list.append(os.path.join(prefix_path_list, video_path_name))

        video_same_path_list.sort()

        final_video_path_Dict[name[:15]] = video_same_path_list

    return final_video_path_Dict

# %%

def get_parameters():

    parser = ArgumentParser()

    # Path
    parser.add_argument('--split_pad_dataset', type=str, default="/workspace/data/split_pad_dataset_512", help="split and pad dataset.")
    parser.add_argument('--three_part_split_pad_dataset', type=str, default="/workspace/data/three_part_split_pad_dataset",
                        help="split split_pad_dataset into three part.")

    return parser.parse_known_args()


# %%
if __name__ == '__main__':

    # for test in jupyter
    parames, unkonwn = get_parameters()

    SPLIT_PAD_DATASET = parames.split_pad_dataset
    SAVE_PATH = parames.three_part_split_pad_dataset
    flag = ['A', 'B', 'C']

    make_folder(SAVE_PATH)

    all_fold = os.listdir(SPLIT_PAD_DATASET)
    all_fold.sort()
    all_fold.remove('raw')

    for fold in all_fold:

        prefix_path_list = get_Diease_Path_List(os.path.join(SPLIT_PAD_DATASET, fold))  # four folder, train/ASD, train/ASD_not, val/ASD, val/ASD_not

        for one in prefix_path_list:
            final_video_path_Dict = get_final_video_path_Dict(one)
            for one_person, p in final_video_path_Dict.items():
                split_num = len(p) // 3

                num = 0

                for i, f in enumerate(flag):

                    if num == split_num * 2:
                        split_num += len(p) - 3 * split_num

                    for save_path in p[num:num+split_num]:
                        saved_path = os.path.join(SAVE_PATH, f, '/'.join(save_path.split('/')[4:]))
                        make_folder('/'.join(saved_path.split('/')[:-1]))
                        shutil.copy(save_path, saved_path)
                        print("copy to ", saved_path)

                    num += split_num

                    print('current person: %s' % one_person)
            
            print('*' * 100)
            print('current path: ', one)


        print('*' * 100)
        print('finish fold ', fold)


