import json
import sys

import numpy as np
from tqdm import tqdm
import argparse
import os

ProjectDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DirProject = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ProjectDir)
sys.path.append(DirProject)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SimpleITK as sitk
import torch
from torch.nn import DataParallel
import pandas as pd
from copy import deepcopy
from monai.inferers import SimpleInferer
import time
import yaml
from torch.utils.data import DataLoader

import CLIP_network
import CLIP_model
from dataloader.CLIP_Dataloader_v2 import CLIP_Dataloader_Both_v2

TrainingFileName = os.path.basename(os.path.abspath(__file__)).split('.')[0]


def add_yaml_params(args_object, yaml_path):
    assert os.path.isfile(yaml_path)
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    for k, v in params.items():
        args_object.__setattr__(k, v)


def infererce_main(topk, save_root, text_include, device):
    global data_iter, clip_model, val_save_dir
    results = []
    df = pd.DataFrame(text_include)
    df.to_excel(os.path.join(save_root, 'text_index.xlsx'))
    prepared_text = []
    for i in tqdm(range(len(text_include))):
        text = [text_include[i][1]]
        with torch.no_grad():
            t_vec = clip_model.text_inference(text)

        prepared_text.append(t_vec)
        save_dir = os.path.join(save_root, 'prepared_text')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{i}.pt')
        torch.save(t_vec, save_path)

    sim_list = []
    count = 0
    error_count = {'count': 0}
    for i in range(1, 11):
        error_count[f'top{i}'] = 0
    for x_image, x_text, info in tqdm(data_iter):
        modal = info['modal'][0]
        name = info['name'][0]

        x_image = x_image.to(device)
        with torch.no_grad():
            image_output, text_output = clip_model(x_image, x_text)

        if not args.see_different_sample:
            save_dir = os.path.join(save_root, f'{name}_{modal}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            print(f'-----------------------------{name}_{modal}_start---------------------------------')
            image_save_path = os.path.join(save_dir, f"{modal}_imageVec.pt")
            torch.save(image_output, image_save_path)
            text_save_path = os.path.join(save_dir, f"{modal}_textVec.pt")
            torch.save(text_output, text_save_path)
            sim = torch.cosine_similarity(image_output, text_output)
            print(f"{name}_{modal}_Cosine_Similarity: {sim.item()}")
            print('-----------------------')
            sim_list.append(sim.item())

            sim_log = []
            for i in range(len(text_include)):
                sim = torch.cosine_similarity(image_output, prepared_text[i])
                sim_log.append({'cosine_sim': sim.item(), 'text': text_include[i][1]})

            sorted_sim_log = sorted(sim_log, key=lambda x: x['cosine_sim'], reverse=True)
            log_df = pd.DataFrame(sorted_sim_log)
            save_path = os.path.join(save_dir, 'sim_log.xlsx')
            log_df.to_excel(save_path)
            txt_path = os.path.join(save_dir, f'TopK_record.txt')

            with open(txt_path, 'w') as f:
                f.write(f'{x_text[0]}\n')
                print(f'{x_text[0]}')
                f.write(f'-------------Top_{topk}---------------------:\n')
                topk_record = {'name': name}
                for tp in range(topk):
                    f.write(f"{sorted_sim_log[tp]['cosine_sim']}   {sorted_sim_log[tp]['text']}\n")
                    print(f"Top_{topk}_{tp} is : {sorted_sim_log[tp]['cosine_sim']}   {sorted_sim_log[tp]['text']}")
                    topk_record[f'top_{tp + 1}'] = sorted_sim_log[tp]['cosine_sim']
                del tp
            results.append({'name': name, 'modal': modal, **topk_record})

            error_count['count'] += 1
            find_flag = False
            for tp in range(10):
                if find_flag:
                    error_count[f'top{tp + 1}'] += 1
                    continue
                if x_text[0] == sorted_sim_log[tp]['text']:
                    error_count[f'top{tp + 1}'] += 1
                    find_flag = True

            print(f'-----------------------------{name}_{modal}_end---------------------------------')
            count += 1
        else:
            pre_image, pre_text = see_batch_sim(image_output, text_output)
            for i in range(len(x_text)):
                print(f'{info["name"][i]} {info["modal"][i]}: {x_text[i]}')
            print("pre_image:")
            print(pre_image)
            text_sim_matrix = get_batch_vectors_sim(text_output)
            text_sim_matrix = text_sim_matrix.numpy().tolist()
            print('text_similarity_matrix:')
            for it in text_sim_matrix:
                print(it)
            print('\n')
            image_sim_matrix = get_batch_vectors_sim(image_output)
            image_sim_matrix = image_sim_matrix.numpy().tolist()
            print('image_similarity_matrix:')
            for it in image_sim_matrix:
                print(it)

            image_output = image_output.cpu().numpy()
            text_output = text_output.cpu().numpy()
            print('-----------------------------------------------------------')

    if not args.see_different_sample:
        print('--------------------------------------------------------------')
        sim_list = np.array(sim_list)
        mean = sim_list.mean()
        print(f'mean similarity is : {mean}')
        print('--------------------------------------------------------------')
        return results, error_count


def see_batch_sim(image_features, text_features):
    image_features = torch.unsqueeze(image_features, dim=0)
    text_features = torch.unsqueeze(text_features, dim=0)

    logits_per_image = image_features @ text_features.transpose(-1, -2)
    logits_per_image = logits_per_image[0]
    logits_per_text = logits_per_image.transpose(-1, -2)
    return logits_per_image, logits_per_text


def get_batch_vectors_sim(vecs):
    b, d = vecs.shape
    ans = torch.zeros((b, b))
    for i in range(b):
        for j in range(i, b):
            v1 = vecs[i, :].reshape(1, d)
            v2 = vecs[j, :].reshape(1, d)
            sim = torch.cosine_similarity(v1, v2)
            ans[i][j] = sim
            ans[j][i] = sim
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=ProjectDir + f'/DATA/',
                        help='directory name which reserves the npy images')
    parser.add_argument('--save_root', type=str, default=ProjectDir + f'/Infer_Results/Pretrain/',
                        help='directory name which reserves the npy images')
    parser.add_argument('--bert_root', '-bert_root', default=f"{ProjectDir}/Text_Model")
    parser.add_argument('--yaml_file_path', '-yaml_file_path',
                        default=f'{ProjectDir}/code/inference/{TrainingFileName}.yaml')
    parser.add_argument('--device', '-device', default='cuda')

    args = parser.parse_args()
    add_yaml_params(args, yaml_path=args.yaml_file_path)
    args.bert_param_path = os.path.join(args.bert_root, args.bert_weight_name)
    if args.ParallelMode:
        args.save_root = os.path.join(ProjectDir, 'Infer_Results', 'TopK_Cosine_Similarity', 'yd_CLIP_Models',
                                      args.Model_Save_Dir)
    else:
        args.save_root = os.path.join(ProjectDir, 'Infer_Results', 'TopK_Cosine_Similarity', args.Model_Save_Dir)

    if hasattr(args, 'save_tag'):
        args.save_root = args.save_root + args.save_tag

    if not args.see_different_sample:
        assert args.batch_size == 1

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.ModelProject in ['inner', 'outer']
    if not args.ParallelMode:
        if args.ModelProject == 'outer':
            snapshot_path = os.path.join(DirProject, 'CLIP_Image_Project_MGPU', args.Model_Save_Dir, args.model_name)
        else:
            snapshot_path = os.path.join(ProjectDir, args.Model_Save_Dir, args.model_name)
    else:
        snapshot_path = os.path.join(DirProject, 'yd_CLIP_Models', args.Model_Save_Dir, args.model_name)

    wait_count = args.wait_count
    for i in range(wait_count):
        time.sleep(1)
        print(f'program will start after {wait_count - i} seconds')

    if args.load_mode != 'model':
        clip_frame = getattr(CLIP_model, args.CLIP_frame)
        clip_model = clip_frame(args)
        clip_model = clip_model.to(args.device)
        param_path = os.path.join(snapshot_path, f'epoch_{args.epoch}.pth')
        weights = torch.load(param_path)['state_dict']
        matching_weights = {k: v for k, v in weights.items() if k in clip_model.state_dict()}
        clip_model.load_state_dict(matching_weights, strict=False)
    else:
        param_path = os.path.join(snapshot_path, f'epoch_{args.epoch}.pth')
        clip_model = torch.load(param_path)
    if args.ParallelMode:
        clip_model = DataParallel(clip_model, device_ids=args.device_ids)

    if hasattr(args, 'bert_attention_mask'):
        clip_model.set_bert_attention_mask(args.bert_attention_mask)

    if hasattr(args, 'return_logit'):
        clip_model.return_logit = args.return_logit
    clip_model.eval()
    clip_model.image_encoder.eval()
    clip_model.text_encoder.eval()
    clip_model.text_pooler.eval()
    val_image_root = os.path.join(args.data_root, 'clip_brain_mri')
    val_save_dir = os.path.join(args.save_root, args.model_name, f'E{args.epoch}')
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    xlsx_save_dir = os.path.join(args.save_root, args.model_name, 'statistic')
    if not os.path.exists(xlsx_save_dir):
        os.makedirs(xlsx_save_dir)
    txt_save_dir = os.path.join(args.save_root, args.model_name, 'error_rate')
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir)

    test_names_path = os.path.join(args.data_root, 'clip_brain_mri')
    test_textVec_root = os.path.join(args.data_root, args.TextVector_Dataset_Dir)

    val_dataset = CLIP_Dataloader_Both_v2(names_root=test_names_path, image_root=val_image_root,
                                       data_transforms=None, text_json_root=test_textVec_root,
                                       image_flip_prob=0, return_val=True, dataset_tag='testing',
                                       image_file_tag='nii.gz', dataset_version=args.dataset_version,
                                       json_version=args.json_version, replace_norm=args.replace_norm,
                                          image_dir_append=args.image_from_dir, configs=args)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, drop_last=False)
    data_iter = iter(val_dataloader)

    res, error_count = infererce_main(topk=args.topk, text_include=val_dataset.text_include, save_root=val_save_dir,
                                      device=args.device)
    try:
        df_a = pd.DataFrame(res)
        mean_row = {'name': 'average'}
        for key in df_a.keys():
            if key == 'name' or key == 'modal':
                continue
            mean_value = df_a[key].mean()
            mean_row[key] = [mean_value]
        mean_df = pd.DataFrame(mean_row)
        print(mean_row)
        df_a = pd.concat([df_a, mean_df])
        write_flag = True
        if os.path.exists(f"{xlsx_save_dir}/E{args.epoch}_CosineSim_TopK.xlsx"):
            con = input(f'{xlsx_save_dir}/E{args.epoch}_CosineSim_TopK.xlsx     is already exist, re-write it?')
            if con == 'yes':
                write_flag = True
            else:
                write_flag = False
        if write_flag:
            df_a.to_excel(f"{xlsx_save_dir}/E{args.epoch}_CosineSim_all_TopK.xlsx")
    except:
        print(f'make all samples xlsx has some wrong!!!')

    try:
        save_path = f"{txt_save_dir}/E{args.epoch}_TopK_error_rate.txt"
        with open(save_path, 'w') as f:
            f.write(f"text count: {error_count['count']}\n")
            for i in range(1, 11):
                error_rate = error_count[f'top{i}'] / error_count['count']
                f.write(f'Top_{i}_error_rate: {error_rate}\n')
    except:
        print(f'make topk error rate txt has some wrong!!!')




