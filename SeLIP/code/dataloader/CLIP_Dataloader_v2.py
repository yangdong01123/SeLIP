import json

import SimpleITK
import torch
from torch.utils.data import Dataset
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from torchvision import transforms
from RandomFlipR import RandomFlipR
from monai.transforms import ToTensor, RandZoom, RandRotate, RandAdjustContrast, RandScaleIntensity
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import SimpleITK as sitk
import logging


class CLIP_Dataloader_Both_v2(Dataset):
    def __init__(self, names_root, image_root, text_json_root, data_transforms, image_flip_prob=0.0,
                 modal_include=('t2_tse_tra', 't1_fl2d_tra', 'ep2d_diff_3scan_trace_p2_ADC',
                        'ep2d_diff_3scan_trace_p2_TRACEW', 't2_tse_tra_p2'), database=(1, 2), dataset_tag='training',
                 image_file_tag='npy', return_val=False, dataset_version=None, json_version='information',
                 replace_norm=None, include_sample=None, word_vector_length=(5000, 5000), return_word_vector=False,
                 device='cpu', image_dir_append='small', configs=None):
        super(CLIP_Dataloader_Both_v2, self).__init__()
        self.return_val = return_val
        if hasattr(configs, 'return_val'):
            self.return_val = configs.return_val
        assert image_file_tag in ('npy', 'nii.gz')
        self.image_file_tag = image_file_tag
        self.dataset_version = dataset_version
        self.replace_norm = replace_norm
        self.word_vector_length = word_vector_length
        self.return_word_vector = return_word_vector
        self.device = device
        self.image_dir_append = image_dir_append
        self.configs = configs
        self.add_sep = False
        if hasattr(configs, 'add_sep'):
            self.add_sep = configs.add_sep

        if hasattr(configs, 'modal_embedding') and configs.modal_embedding == True:
            if not hasattr(configs, 'modal_embedding_code'):
                modal_emb_json = os.path.join(text_json_root, 'modal_embedding_code.json')
            else:
                modal_emb_json = os.path.join(text_json_root, f'{configs.modal_embedding_code}.json')
            logging.info(f'modal embedding code is {modal_emb_json}')
            with open(modal_emb_json, 'r') as file:
                self.modal_embedding_dict = json.load(file)
            logging.info(self.modal_embedding_dict)

        self.names_list = {}
        self.text_json_data = {}

        for base in database:
            json_path = os.path.join(text_json_root, f'text_data_{base}', 'splited_json_data', f'{json_version}.json')
            print(f'json path: {json_path}')
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.text_json_data[base] = data

            txt_path = os.path.join(names_root, f'nii_mri_{base}_data', 'split_txt', f'{dataset_tag}.txt')
            with open(txt_path, 'r') as f:
                names = f.readlines()
            names = [name.replace('\n', '') for name in names]
            if include_sample is not None:
                names = names[:include_sample]
            self.names_list[base] = names

        part_index_path = os.path.join(text_json_root, 'text_data_1', 'static_modal_position_info', 'words_index', 'part_index.xlsx')
        part_df = pd.read_excel(part_index_path)
        part_dict = self.df_to_dict(part_df, 1, 2)
        self.part_index = part_dict
        info_index_path = os.path.join(text_json_root, 'text_data_1', 'static_modal_position_info', 'words_index', 'info_index.xlsx')
        info_df = pd.read_excel(info_index_path)
        info_dict = self.df_to_dict(info_df, 1, 2)
        self.info_index = info_dict

        self.image_root = image_root
        self.data_transforms = data_transforms
        self.modal_include = modal_include
        if hasattr(configs, 'modal_include'):
            self.modal_include = configs.modal_include
        self.image_flip_prob = image_flip_prob
        if image_flip_prob != 0 or image_flip_prob is None:
            self.image_fliper = transforms.Compose([
                RandomFlipR(prob=image_flip_prob)
            ])
            print('have image flip!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            self.image_fliper = None
            print('not have image flip!!!!!!!!!!!!!!!!!!!!!!!')
        self.to_tensor = transforms.Compose([
            ToTensor()
        ])
        self.modal_word_converts = {'t2_tse_tra': 'T2', 't1_fl2d_tra': 'T1',
                                   'ep2d_diff_3scan_trace_p2_ADC': 'ADC', 'ep2d_diff_3scan_trace_p2_TRACEW': 'DWI', 'FLAIR': 'FLAIR'}
        self.image_text_pair = []
        temp_text_include = []
        for base in database:
            image_paths, text_include = self.confirm_data(base)
            self.image_text_pair += image_paths
            temp_text_include += text_include

        temp_text_include = set(temp_text_include)
        temp_text_include = list(temp_text_include)

        self.text_include = [[i, temp_text_include[i]] for i in range(len(temp_text_include))]
        print(len(self.image_text_pair))
        logging.info(f'共有{len(self.image_text_pair)}对配对的图像和文本')
        # sys.exit()

    def df_to_dict(self, df, k, v):
        ans = {}
        df_list = df.values.tolist()
        for item in df_list:
            key = item[k]
            value = item[v]
            ans[key] = value
        return ans


    def confirm_data(self, database):
        check_jdjq = [0, 0]
        modal_count = {}
        show_norm = False
        image_text_pair = []
        all_text = []
        names_database = self.names_list[database]
        database_samples_names = list(self.text_json_data[database].keys())
        image_from_dir = f'nii_mri_{database}_processed'
        if self.image_dir_append != 'None':
            image_from_dir += f'_{self.image_dir_append}'
        for name in tqdm(names_database):
            image_paths = {}
            for modal in self.modal_include:
                if self.dataset_version is None:
                    modal_image_path = os.path.join(self.image_root, f'nii_mri_{database}_data',
                                                    image_from_dir, modal,
                                                    f'{name}.{self.image_file_tag}')
                else:
                    modal_image_path = os.path.join(self.image_root, f'nii_mri_{database}_data', f'Version_{self.dataset_version}',
                                                    image_from_dir, modal,
                                                    f'{name}.{self.image_file_tag}')
                if os.path.exists(modal_image_path):
                    image_paths[modal] = modal_image_path
                else:
                    pass
            if len(image_paths) == 0:
                continue

            if name not in database_samples_names:
                continue
            text_data = self.text_json_data[database][name]
            text_key = text_data.keys()
            text_key = list(text_key)
            normal_record = False
            for key in image_paths.keys():
                converted_key = self.modal_word_converts[key]
                if len(text_key) == 1 and text_key[0] == 'normal':
                    normal_text = text_data[text_key[0]]
                    if self.replace_norm is not None and self.replace_norm != 'None':
                        normal_text = self.replace_norm
                    image_text_pair.append({'name': name, 'image_path': image_paths[key], 'image_modal': converted_key,
                                            'text_NoFlip': normal_text, 'text_Flip': normal_text})
                    if "基底节区" in normal_text:
                        check_jdjq[1] += 1
                    else:
                        check_jdjq[0] += 1
                    if converted_key in modal_count.keys():
                        modal_count[converted_key] += 1
                    else:
                        modal_count[converted_key] = 1
                    if 'normal' in modal_count.keys():
                        if not normal_record:
                            modal_count['normal'] += 1
                    else:
                        modal_count['normal'] = 1
                    normal_record = True
                    if normal_text not in all_text:
                        all_text.append(normal_text)
                    if not show_norm:
                        print(f"normal text is : {normal_text}")
                        show_norm = True
                elif (converted_key in text_key):
                    text = text_data[converted_key]
                    if self.add_sep:
                        text = text.replace("，", "[SEP]")
                    fliped_text = deepcopy(text)
                    fliped_text = fliped_text.replace('左侧', 'toright')
                    fliped_text = fliped_text.replace('右侧', 'toleft')
                    fliped_text = fliped_text.replace('toleft', '左侧')
                    fliped_text = fliped_text.replace('toright', '右侧')
                    image_text_pair.append({'name': name, 'image_path': image_paths[key], 'image_modal': converted_key,
                                            'text_NoFlip': text, 'text_Flip': fliped_text})
                    if "基底节区" in text:
                        check_jdjq[1] += 1
                    else:
                        check_jdjq[0] += 1
                    if converted_key in modal_count.keys():
                        modal_count[converted_key] += 1
                    else:
                        modal_count[converted_key] = 1
                    if text not  in all_text:
                        all_text.append(text)
        print(f'check_jdjq: {check_jdjq}')
        print(f'modal_count: {modal_count}')
        return image_text_pair, all_text

    def percent_crop(self, image_arr, alpha=0.1):
        min_percent = np.percentile(image_arr, alpha)
        max_percent = np.percentile(image_arr, 100 - alpha)
        image_arr[image_arr < min_percent] = min_percent
        image_arr[image_arr > max_percent] = max_percent
        return image_arr

    def get_part_and_info(self, word):
        word_list = word.split('，')
        splited_list = []
        for wd in word_list:
            temp = wd.split('表现为')
            splited_list.append({'part': temp[0].replace('在', ''), 'info': temp[1]})
        return splited_list

    def make_word_vector(self, word_dict, device):
        part_vector = torch.zeros((1, self.word_vector_length[0]), device=device)
        info_vector = torch.zeros((1, self.word_vector_length[1]), device=device)

        for wd_d in word_dict:
            part = wd_d['part']
            info = wd_d['info']
            if part not in self.part_index.keys():
                part_vector[0][0] = 1
            else:
                part_vector[0][self.part_index[part]] = 1

            if info not in self.info_index.keys():
                info_vector[0][0] = 1
            else:
                info_vector[0][self.info_index[info]] = 1
        final_vector = torch.cat([part_vector, info_vector], dim=-1)
        return final_vector

    def replace_in_text(self, word, set_info=None,
                        normal_info="头颅形态、大小未见异常。脑实质未见异常信号影，所见脑室形态及脑沟裂池未见异常扩张或变窄，中线结构未见移位。"):
        if word == normal_info:
            refined_text = word
        else:
            sents = word.split("，")
            refined_text = ""
            for se in sents:
                part1, part2 = se.split("表现为")
                part1 = part1 + "表现为"
                if set_info is not None:
                    refined_Se = part1 + set_info
                else:
                    if "高信号" in part2:
                        refined_Se = part1 + "高信号"
                    elif "降低" in part2:
                        refined_Se = part1 + "低信号"
                    elif "减少" in part2:
                        refined_Se = part1 + "低信号"
                    elif "短信号" in part2:
                        refined_Se = part1 + "短信号"
                    elif "等信号" in part2:
                        refined_Se = part1 + "等信号"
                    elif "长信号" in part2:
                        refined_Se = part1 + "长信号"
                    else:
                        refined_Se = part1 + part2
                refined_text = refined_text + "，" + refined_Se
            refined_text = refined_text.replace("，", "", 1)
        return refined_text

    def info_replace_semantic(self, text_list=None, text=None,
                              normal_info="头颅形态、大小未见异常。脑实质未见异常信号影，所见脑室形态及脑沟裂池未见异常扩张或变窄，中线结构未见移位。",
                              set_info=None):
        assert text_list is not None or text is not None
        if text_list is not None:
            refined_list = []
            for item in text_list:
                refined_text = self.replace_in_text(item, normal_info=normal_info, set_info=set_info)
                refined_list.append(refined_text)
            return refined_list
        else:
            refined_text = self.replace_in_text(text, normal_info=normal_info, set_info=set_info)
            return refined_text

    def __getitem__(self, index):
        sample_pair = self.image_text_pair[index]

        if self.image_file_tag == 'npy':
            x_image = np.load(sample_pair['image_path'])
        else:
            image = sitk.ReadImage(sample_pair['image_path'])
            x_image = sitk.GetArrayFromImage(image)
        x_image = self.percent_crop(x_image)
        x_image = x_image.astype(np.float32)
        x_image = np.expand_dims(x_image, axis=0)
        x_image = np.ascontiguousarray(x_image)

        text_clip_flag = False
        if self.data_transforms is not None:
            x_image = self.data_transforms(x_image)
            if self.image_flip_prob != 0:
                x_image, flip_count = self.image_fliper(x_image)
                flag = flip_count % 2
                if flag != 0:
                    text_clip_flag = True
            x_image = self.to_tensor(x_image)

        if text_clip_flag:
            x_text = sample_pair['text_Flip']
        else:
            x_text = sample_pair['text_NoFlip']

        if hasattr(self.configs, 'replace_info_semantic') and self.configs.replace_info_semantic == True:
            if hasattr(self.configs, 'set_info'):
                x_text = self.replace_in_text(x_text, set_info=self.configs.set_info, normal_info=self.replace_norm)
            else:
                x_text = self.replace_in_text(x_text, normal_info=self.replace_norm)

        if self.return_word_vector:
            if x_text == self.replace_norm:
                word_vector = torch.zeros((1, self.word_vector_length[0] + self.word_vector_length[1]),
                                          device=self.device)
            else:
                word_dict = self.get_part_and_info(x_text)
                word_vector = self.make_word_vector(word_dict, self.device)

        x_image = (x_image - x_image.min()) / (x_image.max() - x_image.min())

        if not self.return_val:
            if not self.return_word_vector:
                return x_image, x_text
            else:
                return x_image, x_text, word_vector
        else:
            return x_image, x_text, {'modal': sample_pair['image_modal'], 'name': sample_pair['name']}

    def __len__(self):
        return len(self.image_text_pair)


if __name__=="__main__":
    pass




