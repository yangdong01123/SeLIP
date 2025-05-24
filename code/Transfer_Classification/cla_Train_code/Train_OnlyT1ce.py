import argparse
import os.path
import logging
import sys
import os

ProjectDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ProjectDir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import yaml
import shutil
import numpy as np
import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.transforms import Rand3DElasticd, RandAdjustContrastd, RandScaleIntensityd, \
    RandCropByPosNegLabeld, RandZoomd, RandFlipd, ToTensord, RandSpatialCrop, RandZoom, \
    RandRotate, RandFlip, Rand3DElastic, RandAdjustContrast, RandScaleIntensity, ToTensor, RandRotated, SpatialPadd, \
    RandSpatialCropd, SpatialPad

import math

import torch
from monai.losses import DiceCELoss, DiceFocalLoss

from torch.optim.lr_scheduler import LambdaLR
from torch.nn import L1Loss, BCELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.cuda import amp

import Transfer_Classification.cla_network as cla_network
from Transfer_Classification.classification_dataset.Classification_dataset import SingleModal_Dataset_Single, SingleModal_Dataset
from Validation_functions import validation_model_modal_embedding_ReturnAcc

TrainingFileName = os.path.basename(os.path.abspath(__file__)).split('.')[0]


def main():
    ce_losser = CrossEntropyLoss()

    labeled_transform = transforms.Compose([
        Rand3DElastic(mode='bilinear', prob=0.3, sigma_range=(0.005, 0.01), magnitude_range=(0.005, 0.01)),
        RandZoom(min_zoom=(1, 0.8, 0.8), max_zoom=(1, 1.2, 1.2), mode='bilinear', prob=0.3),
        RandRotate(range_x=0.53, range_y=0.0, range_z=0.0,
                   mode='bilinear',
                   prob=0.3, padding_mode="zeros"),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandFlip(spatial_axis=2, prob=0.5),
        RandAdjustContrast(prob=0.2, gamma=(0.9, 1.1)),
        RandScaleIntensity(0.1, prob=0.2),
        ToTensor(),
    ])

    lgg_path = f"{ProjectDir}/DATA/classification/BraTs2019/MICCAI_BraTS_2019_Data_Training/name_dir/sub_train_set/LGG_train_sub_0p{args.sub_num}.txt"
    hgg_path = f"{ProjectDir}/DATA/classification/BraTs2019/MICCAI_BraTS_2019_Data_Training/name_dir/sub_train_set/HGG_train_sub_0p{args.sub_num}.txt"

    val_lgg_path = f"{ProjectDir}/DATA/classification/BraTs2019/MICCAI_BraTS_2019_Data_Training/name_dir/LGG_test_samples.txt"
    val_hgg_path = f"{ProjectDir}/DATA/classification/BraTs2019/MICCAI_BraTS_2019_Data_Training/name_dir/HGG_test_samples.txt"

    logging.info(lgg_path)
    logging.info(hgg_path)
    logging.info(val_lgg_path)
    logging.info(val_hgg_path)

    labeled_data_root = os.path.join(args.data_root, 'NPY_Dataset')

    lgg_batchsize = int(args.batch_size / 2)
    hgg_batchsize = args.batch_size - lgg_batchsize
    lgg_datasets_labeled = SingleModal_Dataset_Single(names_path_lgg=lgg_path, names_path_hgg=None,
                                                     data_root=labeled_data_root,
                                                     transforms=labeled_transform, cache=False,
                                                     input_tag='t1ce',
                                                     sub_sample_num=args.sub_sample_num,
                                                     need_num=lgg_batchsize * args.ITERLENGTH, return_info=False)
    lgg_dataloader_labeled = DataLoader(lgg_datasets_labeled, batch_size=lgg_batchsize, shuffle=True,
                                        pin_memory=True, drop_last=True, num_workers=args.load_bs)
    lgg_labeled_iter = iter(lgg_dataloader_labeled)

    hgg_datasets_labeled = SingleModal_Dataset_Single(names_path_lgg=None, names_path_hgg=hgg_path,
                                                     data_root=labeled_data_root,
                                                     transforms=labeled_transform, cache=False,
                                                     input_tag='t1ce',
                                                     sub_sample_num=args.sub_sample_num,
                                                     need_num=hgg_batchsize * args.ITERLENGTH, return_info=False)
    hgg_dataloader_labeled = DataLoader(hgg_datasets_labeled, batch_size=hgg_batchsize, shuffle=True,
                                        pin_memory=True, drop_last=True, num_workers=args.load_bs)
    hgg_labeled_iter = iter(hgg_dataloader_labeled)

    if args.Mix_Precision:
        scaler = amp.GradScaler()

    global cla_model
    optim = SGD(cla_model.parameters(), lr=args.image_lr, momentum=0.9)
    warmup_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: min(1.0, epoch / args.WARMUP_ITER))
    training_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / args.EPOCH) ** 0.9)

    if args.resume is not None:
        resume_path = os.path.join(snapshot_path, f'epoch_{args.resume_epoch}.pth')
        checkpoint = torch.load(resume_path)
        cla_model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optim_dict'])
        cla_model.image_encoder.train()
        msg = f'resume from: {args.resume}, resume_epoch: {args.resume_epoch}\n'
        logging.info(msg)

    loss_record = []
    cla_model.image_encoder.train()
    if args.resume is not None:
        start_epoch = args.resume_epoch + 1
    else:
        start_epoch = 0

    if args.WARMUP:
        start_epoch -= 1
    cla_model.train()

    best_acc = {'acc': 0, 'epoch': '-1'}
    for epoch in range(start_epoch, args.EPOCH):
        time_epoch_start = time.time()
        iter_nums = args.ITERLENGTH
        if args.WARMUP and epoch == -1:
            iter_nums = args.WARMUP_ITER
        for iteration in range(iter_nums):
            torch.cuda.synchronize()
            time_iter_start = time.time()
            if args.WARMUP and epoch == -1:
                warmup_scheduler.step()

            try:
                x_lgg, y_lgg = next(lgg_labeled_iter)
            except:
                lgg_labeled_iter = iter(lgg_dataloader_labeled)
                x_lgg, y_lgg = next((lgg_labeled_iter))

            try:
                x_hgg, y_hgg = next(hgg_labeled_iter)
            except:
                hgg_labeled_iter = iter(hgg_dataloader_labeled)
                x_hgg, y_hgg = next((hgg_labeled_iter))

            x = torch.cat([x_lgg, x_hgg], dim=0)
            target = torch.cat([y_lgg, y_hgg], dim=0)

            x = x.cuda()
            target = target.cuda()
            input_data = {'x': x}
            if args.Mix_Precision:
                optim.zero_grad()
                with amp.autocast():
                    output = cla_model(**input_data)
                    loss = ce_losser(output, target)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                output = cla_model(**input_data)
                loss = ce_losser(output, target)
                optim.zero_grad()
                loss.backward()
                optim.step()

            loss_record.append(loss.item())

            torch.cuda.synchronize()
            time_iter_end = time.time()

            if loss.item() > 100000000:
                print('model grad vanish, exit system!')
                sys.exit()
            else:
                pass

            logging.info(f"epoch {epoch}: {iteration + 1}/{args.ITERLENGTH}, loss:{loss.item():.3f}, "
                         f"time:{(time_iter_end - time_iter_start): .3f}, lr:{optim.state_dict()['param_groups'][0]['lr']:.6f}")

        time_epoch_end = time.time()
        logging.info(f"TRAIN EPOCH {epoch}, lr: {optim.state_dict()['param_groups'][0]['lr']}, "
                     f"time: {(time_epoch_end - time_epoch_start):.3f}")

        plt_losses(loss_record, epoch)
        if args.WARMUP and epoch == -1:
            pass
        else:
            training_scheduler.step()

        if ((epoch + 1) % args.Checkpoint_Save_Space == 0):
            model_save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
            torch.save(cla_model, model_save_path)
            optim_save_path = os.path.join(snapshot_path, f'optim_{epoch}.pth')
            torch.save({'optim_dict': optim.state_dict()}, optim_save_path)

        '''
                validation part
                '''
        Validation_flag = False
        if (epoch == 0) or ((epoch + 1) % args.VALIDATION_SPACE == 0) or (epoch == 4):
            Validation_flag = True
        if Validation_flag:
            logging.info('********* validation now *********')
            cla_model.eval()

            val_save_root = f'{ProjectDir}/Validation_View_TransferSeg/{args.ValidationNII_Dir}/{args.model_name}'
            # val_data_root = os.path.join(args.data_root, 'normed_resized')
            val_dataset = SingleModal_Dataset(names_path_lgg=val_lgg_path, names_path_hgg=val_hgg_path,
                                             data_root=labeled_data_root,
                                             transforms=None, cache=False,
                                             input_tag='t1ce', return_info=True)

            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                        pin_memory=True, drop_last=True)
            if os.path.exists(val_save_root) == False:
                os.makedirs(val_save_root)

            acc = validation_model_modal_embedding_ReturnAcc(model=cla_model, val_dataloader=val_dataloader, save_dir=val_save_root,
                                             epoch=epoch)

            if acc > best_acc['acc']:
                best_acc['acc'] = acc
                best_acc['epoch'] = epoch
                best_model_save_path = os.path.join(snapshot_path, f'best_model.pth')
                torch.save(cla_model, best_model_save_path)
                best_optim_save_path = os.path.join(snapshot_path, f'best_optim.pth')
                torch.save({'optim_dict': optim.state_dict()}, best_optim_save_path)

                logging.info(f'best model update: acc_{acc:.4f}')

            cla_model.train()
            logging.info('********* validation end *********')

    model_save_path = os.path.join(snapshot_path, f'epoch_{args.EPOCH - 1}.pth')
    if not os.path.exists(model_save_path):
        torch.save(cla_model, model_save_path)
        optim_save_path = os.path.join(snapshot_path, f'optim_{args.EPOCH - 1}.pth')
        torch.save({'optim_dict': optim.state_dict()}, optim_save_path)
    logging.info(f"best info: acc_{best_acc['acc']}  epoch_{best_acc['epoch']}")


def check_and_backup():
    if args.resume != None:
        # 确认一下是否继续训练，并确认继续训练的模型地址
        resume_path = os.path.join(ProjectDir, 'models', args.model_name, args.resume)
        print(f'******resume path is: {resume_path}******\n')
        resume_flag = input('do you want to continue resume action?')
        if resume_flag not in {'Y', 'y'} or resume_flag.lower() != 'yes':
            sys.exit()
    else:
        # 如果为训练新的模型，为了防止忘记修改信息而将训练好的文件覆盖，此处会检测文件夹名称，若存在则自动退出
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
            print(f"创建{snapshot_path}")
        else:
            print(f"{snapshot_path} is exist, the program will exit!!!")
            sys.exit()

        # 将训练时的代码备份到保存model的文件夹下
        if os.path.exists(snapshot_path + f'/code'):
            if_delete = input(f"是否移除{snapshot_path + '/code'}?")
            if if_delete == "y" or if_delete == "Y":
                print(f"移除{snapshot_path + f'{os.sep}code'}")
                shutil.rmtree(snapshot_path + f'{os.sep}code')
            else:
                sys.exit()
        print("---------------copy code--------------------")
        shutil.copytree(ProjectDir + f"/code", snapshot_path + f"/code")
        # 单独一个文件夹用来备份训练各个模型时的代码
        if os.path.isdir(ProjectDir + f"/previous_code/{args.model_name}_code"):
            print(f"移除{snapshot_path + f'{os.sep}code'}，重新进行拷贝")
            shutil.rmtree(ProjectDir + f"/previous_code/{args.model_name}_code")

        shutil.copytree(ProjectDir + f"/code", ProjectDir + f"/previous_code/{args.model_name}_code")


def plt_losses(loss_list, epoch):
    plt.cla()
    plt.plot(range(len(loss_list)), loss_list, color='blue', label='Generator')
    if os.path.isdir(snapshot_path + f"/loss_version"):
        plt.savefig(snapshot_path + f"/loss_version/epoch_{epoch}.jpg")
    else:
        os.makedirs(snapshot_path + f"/loss_version")
        plt.savefig(snapshot_path + f"/loss_version/epoch_{epoch}.jpg")


def adjust_lr(optimizer, epoch, num_epoch, power, init_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * np.power((1 - epoch / num_epoch), power)


def warmup_lr(optimizer, iter, num_iter, init_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * (iter / num_iter)
        print(param_group['lr'])


def add_yaml_params(args_object, yaml_path):
    assert os.path.isfile(yaml_path)
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    for k, v in params.items():
        args_object.__setattr__(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(ProjectDir, 'DATA', 'classification', 'BraTs2019', 'MICCAI_BraTS_2019_Data_Training'),
                        help='directory name which reserves the npy images')
    parser.add_argument('--yaml_file_path', '-yaml_file_path',
                        default=f'{ProjectDir}/code/Transfer_Classfication/cla_Train_yaml/{TrainingFileName}.yaml')
    parser.add_argument('--resume', '-resume', default=None)
    parser.add_argument('--name_start', '-name_start', default='TFS')
    parser.add_argument('--sub_num', '-sub_num', default=1)
    parser.add_argument('--set_hidden_layer', '-set_hidden_layer', default=None)
    parser.add_argument('--name_end', '-name_end', default=None)

    args = parser.parse_args()
    add_yaml_params(args, yaml_path=args.yaml_file_path)

    if args.set_hidden_layer is not None:
        args.model['hidden_layer'] = args.set_hidden_layer

    if args.name_start == 'TFS':
        args.Use_Pretrain = False
        args.frozen_encoder = False
    elif args.name_start == 'FT':
        args.Use_Pretrain = True
        args.frozen_encoder = False
    else:
        args.Use_Pretrain = True
        args.frozen_encoder = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    assert args.model['hidden_layer'] is None or args.model['hidden_layer'] in ['None', 'Linear', 'Conv']
    if args.model['hidden_layer'] is None or args.model['hidden_layer'] == 'None':
        args.model_name = f'NoHidden_{args.name_start}_Train0p{args.sub_num}'
    elif args.model['hidden_layer'] == 'Conv':
        args.model_name = f'ConvHidden_{args.name_start}_Train0p{args.sub_num}'
    else:
        args.model_name = f'{args.name_start}_Train0p{args.sub_num}'

    if args.name_end is not None:
        args.model_name = f'{args.model_name}_{args.name_end}'

    pretrain_model_name = os.path.basename(os.path.dirname(args.Pretrain_path))
    snapshot_path = os.path.join(ProjectDir, args.Save_Dir, pretrain_model_name, args.model_name)

    cla_frame = getattr(cla_network, args.Seg_frame)
    cla_model = cla_frame(args)
    cla_model = cla_model.to(args.device)

    frozen_layers = ['image_encoder']
    if args.Use_Pretrain:
        pretrain_model_path = os.path.join(ProjectDir, args.Pretrain_path)
        pretrain_model = torch.load(pretrain_model_path)
        weights = pretrain_model
        matching_weights = {}
        for k, v in weights.items():
            if k in cla_model.state_dict():
                matching_weights[k] = v
                print(f'load: {k}')
        cla_model.load_state_dict(matching_weights, strict=False)
        del pretrain_model, weights, matching_weights
        print(
            '-----------------------------------------------------------------------------------------------------------------------------')

    if hasattr(args, 'frozen_encoder') and args.frozen_encoder:
        for k, v in cla_model.named_parameters():
            if (k.split('.')[0] in frozen_layers) or (k.split('.')[0] + '.' + k.split('.')[1] in frozen_layers):
                v.requires_grad = False
                print(f'{k} requires_grad is False')
            else:
                print(f'{k} requires_grad is True')

    if args.resume is not None:
        assert args.resume_epoch is not None
    else:
        check_and_backup()
        pass

    # log模块的设定
    logging.basicConfig(filename=snapshot_path + f"/log.txt", level=logging.INFO,
                        filemode='a', format='[%(asctime)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(os.path.basename(os.path.abspath(__file__)))
    logger = logging.getLogger()
    logging.info(args.introduction)
    if args.Use_Pretrain:
        logging.info('USING PRETRAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    with open(os.path.join(snapshot_path, 'README.txt'), 'w') as f:
        f.write(args.introduction)

    split_args = str(args).replace('Namespace(', '')[:-1].split(',')
    for arg in split_args:
        logging.info(arg)

    wait_count = args.wait_count
    for i in range(wait_count):
        time.sleep(1)
        print(f'program will start after {wait_count - i} seconds')
    main()