'''
main entrance of prepare video.
'''

# %%
import os
import sys

# there should exchange the path with yourself path.
sys.path.append('/workspace/Walk_Video_PyTorch/project')

import warnings
warnings.filterwarnings("ignore")

from utils import make_folder, count_File_Number
from argparse import ArgumentParser
from torchvision.io import read_video, read_video_timestamps, write_video
import torch
#from batch_detection import Batch_Detection
# %%

def make_split_pad_folder(split_pad_data_path: str):
    '''
    make the split pad video folder, for /train/ASD; /train/ASD_not;, /val/ASD; /val/ASD_not

    Args:
        split_pad_data_path (str): '/workspace/data/split_pad_dataset'
    '''

    for flag in ('train', 'val'):

        data_path_flag = os.path.join(split_pad_data_path, flag)

        for diease_flag in ('ASD', 'ASD_not'):

            final_split_pad_data_path = os.path.join(data_path_flag, diease_flag)

            make_folder(final_split_pad_data_path)

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

    for data_path_flag_diease in prefix_path_list:

        compare_video_name_list = []

        # all video path
        video_path_list = os.listdir(data_path_flag_diease)

        # start compare video path list
        for compare_video_name in video_path_list:

            compare_video_name_list.append(compare_video_name[:15])

        compare_video_file_name = list(set(compare_video_name_list))

        for name in compare_video_file_name:

            video_same_path_list = []

            for video_path_name in video_path_list:

                now_video_path_name = video_path_name[:15]

                if now_video_path_name in name:

                    # store the full path of unique data with diease in a list.
                    video_same_path_list.append(os.path.join(data_path_flag_diease, video_path_name))

            video_same_path_list.sort()

            final_video_path_Dict[os.path.join(data_path_flag_diease, name[:15])] = video_same_path_list

    return final_video_path_Dict

# %%


def read_and_write_video_from_List(path_list: list, img_size: int = 256, video_save_path: str = '', flag: str = 'pad'):

    # instance batch detection class.
    get_bbox = Batch_Detection(img_size=img_size)

    for path in path_list:

        after_path = path[24:]
        save_video_path = os.path.join(video_save_path, after_path)

        path_split_file_name = path.split('/')[-1]
        video_save_path_file_name = save_video_path.split('/')[-1]

        # if path_split_file_name  != video_save_path_file_name:

        video_frame, audio_frame, video_info = read_video(path)  # (t, h, w, c)
        print("currect file:", path)

        clip_pad_imgs = get_bbox.handel_batch_imgs(video_frame, flag=flag)  # c, t, h, w

        write_video(filename=save_video_path, video_array=clip_pad_imgs.permute(
            1, 2, 3, 0), fps=30, video_codec='h264')  # (c, t, h, w) to (t, h, w, c)

        print(video_info['video_fps'], path, video_frame.shape)
        print('clip %i frames!, %i frams lost!' %
              (clip_pad_imgs.size()[1], video_frame.size()[0] - clip_pad_imgs.size()[1]))
        print('----------------------------------------------------------------------')

    return video_frame, path, video_info['video_fps']


# %%

def get_parameters():

    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--img_size', type=int, default=512)

    # Training setting
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader for load video')

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/dataset/", help='meta dataset path')
    parser.add_argument('--split_pad_data_path', type=str, default="/workspace/data/split_pad_dataset",
                        help="split and pad dataset with detection method.")
    parser.add_argument('--split_data_path', type=str, default="/workspace/data/split_dataset",
                        help="split dataset with detection method.")
    parser.add_argument('--pad_flag', type=str, default="pad", help="flag that pad or not")

    return parser.parse_known_args()


# %%
if __name__ == '__main__':

    # for test in jupyter
    parames, unkonwn = get_parameters()

    DATA_PATH = parames.data_path
    IMG_SIZE = parames.img_size
    PAD_FLAG = parames.pad_flag

    # make folder with img size
    if PAD_FLAG == 'pad':
        SPLIT_PAD_DATA_PATH = parames.split_pad_data_path + "_" + str(IMG_SIZE)
        make_split_pad_folder(SPLIT_PAD_DATA_PATH)
    else:
        SPLIT_DATA_PATH = parames.split_data_path + "_" + str(IMG_SIZE)
        make_split_pad_folder(SPLIT_DATA_PATH)

    prefix_path_list = get_Diease_Path_List(DATA_PATH)  # four folder, train/ASD, train/ASD_not, val/ASD, val/ASD_not

    final_video_path_Dict = get_final_video_path_Dict(prefix_path_list)

    for key in final_video_path_Dict.keys():

        now_video_path_list = final_video_path_Dict[key]
        now_video_path_list.sort()

        if PAD_FLAG == 'pad':
            video_frame, now_video_path_list, video_info = read_and_write_video_from_List(
                now_video_path_list, img_size=IMG_SIZE, video_save_path=SPLIT_PAD_DATA_PATH, flag=PAD_FLAG)
        else:
            video_frame, now_video_path_list, video_info = read_and_write_video_from_List(
                now_video_path_list, img_size=IMG_SIZE, video_save_path=SPLIT_DATA_PATH, flag=PAD_FLAG)

    print('Finish split and pad video!')

    count_File_Number(SPLIT_DATA_PATH)
    count_File_Number(SPLIT_PAD_DATA_PATH)


# %%
# merge differenct videos into one video.

# splt_video_path = "/workspace/Walk_Video_PyTorch/project/prepare_video/test"

# splt_video_path_list = os.listdir(splt_video_path)
# #%%
# video_list = []

# for path in splt_video_path_list:
#     video, audio, video_info = read_video(os.path.join(splt_video_path, path))
#     video_list.append(video)

# video_file = torch.cat(video_list, dim=(0))

# # %%

# write_video(filename=os.path.join(splt_video_path, 'full.mp4'), fps=30, video_codec='h264', video_array=video_file)
