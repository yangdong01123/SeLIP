from torch.utils.data import Dataset
import os
import numpy as np
import torch
from einops import rearrange


class SingleModal_Dataset(Dataset):
    def __init__(self, names_path_lgg, names_path_hgg, data_root, input_tag, transforms=None,
                 cache=False, sub_sample_num='None', file_tag='npy', return_info=False, need_num=None):
        self.return_info = return_info
        samples_dict = []

        with open(names_path_lgg, 'r') as f:
            names = f.readlines()
            names = [name.replace('\n', '') for name in names]
        if sub_sample_num != 'None':
            assert sub_sample_num['lgg'] <= len(names)
            names = names[:sub_sample_num['lgg']]
        print(f'names number is :{len(names)}')
        for i in range(len(names)):
            if names[i] == '\n':
                continue

            paths_list = []
            input_img_path = os.path.join(data_root, 'LGG', names[i], f'{names[i]}_{input_tag}.{file_tag}')
            paths_list.append(input_img_path)

            samples_dict.append({'image_path': paths_list, 'name': names[i], 'label': 0, 'mode': 'LGG'})

        with open(names_path_hgg, 'r') as f:
            names = f.readlines()
            names = [name.replace('\n', '') for name in names]
        if sub_sample_num != 'None':
            assert sub_sample_num['hgg'] <= len(names)
            names = names[:sub_sample_num['hgg']]
        print(f'names number is :{len(names)}')

        for i in range(len(names)):
            if names[i] == '\n':
                continue

            paths_list = []
            input_img_path = os.path.join(data_root, 'HGG', names[i], f'{names[i]}_{input_tag}.npy')
            paths_list.append(input_img_path)

            samples_dict.append({'image_path': paths_list, 'name': names[i], 'label': 1, 'mode': 'HGG'})

        if need_num is not None:
            repeat_num = int(need_num / len(samples_dict)) + 1
            samples_dict = samples_dict * repeat_num
        self.samples_dict = samples_dict
        self.transforms = transforms
        self.cache = cache

    def __getitem__(self, index):
        sample = self.samples_dict[index]

        paths = sample['image_path']

        x_list = []
        for pth in paths:
            tp = np.load(pth)
            x_list.append(tp)
        x = np.array(x_list)
        x = x.astype(np.float32)
        # x = np.expand_dims(x, axis=0)
        x = np.ascontiguousarray(x)

        if self.transforms:
            x = self.transforms(x)

        if self.return_info:
            return x, sample['label'], sample
        else:
            return x, sample['label']


    def __len__(self):
        return len(self.samples_dict)


class SingleModal_Dataset_Single(Dataset):
    def __init__(self, names_path_lgg, names_path_hgg, data_root, input_tag, transforms=None,
                 cache=False, sub_sample_num='None', file_tag='npy', return_info=False, need_num=None):
        self.return_info = return_info
        samples_dict = []

        assert names_path_lgg is None or names_path_hgg is None

        if names_path_lgg is not None:
            with open(names_path_lgg, 'r') as f:
                names = f.readlines()
                names = [name.replace('\n', '') for name in names]
            if sub_sample_num != 'None':
                assert sub_sample_num['lgg'] <= len(names)
                names = names[:sub_sample_num['lgg']]
            print(f'names number is :{len(names)}')

            for i in range(len(names)):
                if names[i] == '\n':
                    continue

                paths_list = []
                input_img_path = os.path.join(data_root, 'LGG', names[i], f'{names[i]}_{input_tag}.{file_tag}')
                paths_list.append(input_img_path)

                samples_dict.append({'image_path': paths_list, 'name': names[i], 'label': 0, 'mode': 'LGG'})
        else:
            with open(names_path_hgg, 'r') as f:
                names = f.readlines()
                names = [name.replace('\n', '') for name in names]
            if sub_sample_num != 'None':
                assert sub_sample_num['hgg'] <= len(names)
                names = names[:sub_sample_num['hgg']]
            print(f'names number is :{len(names)}')

            for i in range(len(names)):
                if names[i] == '\n':
                    continue

                paths_list = []

                input_img_path = os.path.join(data_root, 'HGG', names[i], f'{names[i]}_{input_tag}.npy')
                paths_list.append(input_img_path)

                samples_dict.append({'image_path': paths_list, 'name': names[i], 'label': 1, 'mode': 'HGG'})


        if need_num is not None:
            repeat_num = int(need_num / len(samples_dict)) + 1
            samples_dict = samples_dict * repeat_num
        self.samples_dict = samples_dict
        self.transforms = transforms
        self.cache = cache

    def __getitem__(self, index):
        sample = self.samples_dict[index]

        paths = sample['image_path']

        x_list = []
        for pth in paths:
            tp = np.load(pth)
            x_list.append(tp)
        x = np.array(x_list)
        x = x.astype(np.float32)
        x = np.ascontiguousarray(x)

        if self.transforms:
            x = self.transforms(x)

        if self.return_info:
            return x, sample['label'], sample
        else:
            return x, sample['label']


    def __len__(self):
        return len(self.samples_dict)


class pop_from_list(object):
    def __call__(self, sample):
        return sample[0]