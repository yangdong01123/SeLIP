import argparse
import os.path
import logging
import sys
import os

ProjectDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ProjectDir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import shutil
import numpy as np
import time
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.transforms import RandAdjustContrast, RandScaleIntensity, \
    RandCropByPosNegLabel, RandZoom, RandFlip, ToTensor, RandRotate, RandAxisFlip, RandSpatialCrop, SpatialPad, \
    Rand3DElastic, RandCoarseDropout
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.cuda import amp

from dataloader.CLIP_Dataloader_v2 import CLIP_Dataloader_Both_v2
import CLIP_model
from Train_code.Validation_functions import validation_Both_TopK

TrainingFileName = os.path.basename(os.path.abspath(__file__)).split('.')[0]
KEYS = ['input', 'target']


def main():
    data_transforms = transforms.Compose([
        RandZoom(min_zoom=(1, 0.9, 0.9), max_zoom=(1, 1.1, 1.1),
                 mode='trilinear', align_corners=True, padding_mode='minimum', prob=0.2),
        RandRotate(range_x=0.53, range_y=0.0, mode='bilinear',
                   prob=0.2, padding_mode="zeros"),
        RandAdjustContrast(prob=0.2, gamma=(0.9, 1.1)),
        RandScaleIntensity(factors=0.1, prob=0.2),
        Rand3DElastic(sigma_range=(0.01, 0.01), magnitude_range=(0.01, 0.01), mode='trilinear', prob=0.1),

        # RandZoom(min_zoom=(1, 0.9, 0.9), max_zoom=(1, 1.1, 1.1),
        #          mode='trilinear', align_corners=True, padding_mode='minimum', prob=1),
        # RandRotate(range_x=0.53, range_y=0.0, mode='bilinear',
        #            prob=1, padding_mode="zeros"),
        # RandAdjustContrast(prob=1, gamma=(0.9, 1.1)),
        # RandScaleIntensity(factors=0.1, prob=1),
        # Rand3DElastic(sigma_range=(0.01, 0.01), magnitude_range=(0.01, 0.01), mode='trilinear', prob=1),

    ])

    train_image_root = os.path.join(args.data_root, args.Image_Dataset_Dir)
    train_textVec_root = os.path.join(args.data_root, args.TextVector_Dataset_Dir)

    train_names_path = os.path.join(args.data_root, 'clip_brain_mri')
    logging.info('train_names_path: ' + train_names_path)
    logging.info('train_image_root: ' + train_image_root)
    logging.info('train_textVec_root: ' + train_textVec_root)
    # with open(train_names_path, 'r') as f:
    #     SAMPLE_NUM = len(f.readlines())
    # print(f'sample num is {SAMPLE_NUM}')
    # repeat_num = math.ceil((args.ITERLENGTH * args.batch_size) / SAMPLE_NUM)

    train_dataset = CLIP_Dataloader_Both_v2(names_root=train_names_path, image_root=train_image_root,
                                            data_transforms=data_transforms, text_json_root=train_textVec_root,
                                            image_flip_prob=0, dataset_tag='training',
                                            dataset_version=args.dataset_version, json_version=args.json_version,
                                            replace_norm=args.replace_norm, return_word_vector=args.return_word_vector,
                                            image_dir_append=args.image_from_dir, configs=args, database=args.databases)
    bs = args.load_bs
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=bs)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, pin_memory=True,
    #                               drop_last=True)
    train_data_iter = iter(train_dataloader)
    if args.Mix_Precision:
        scaler = amp.GradScaler()

    # learning_rate_fns = [
    #     lambda epoch: args.image_lr * (1 - epoch / args.EPOCH) ** 2,  # image的学习率衰减函数
    #     lambda epoch: args.text_lr * (1 - epoch / args.EPOCH) ** 2,  # text的学习率衰减函数
    # ]

    global clip_model
    parameters = [
        {'params': clip_model.image_encoder.parameters(), 'lr': args.image_lr},
        # {'params': clip_model.logit_scale.parameters(), 'lr': args.image_lr},
        {'params': clip_model.text_encoder.parameters(), 'lr': args.text_lr},
        {'params': clip_model.text_pooler.parameters(), 'lr': args.text_lr}
    ]
    WEIGHT_DECAY = 0
    if hasattr(args, 'weight_decay'):
        WEIGHT_DECAY = args.weight_decay
    logging.info(f'WEIGHT_DECAY is : {WEIGHT_DECAY}')

    eps = 1e-8
    if hasattr(args, 'eps'):
        eps = args.eps
    logging.info(f'eps is : {eps}')
    optim = Adam(parameters, lr=args.image_lr, weight_decay=WEIGHT_DECAY, eps=eps)
    if args.resume is not None:
        resume_optim_path = os.path.join(snapshot_path, f'optim_{args.resume_epoch}.pth')
        optim_param = torch.load(resume_optim_path)
        optim.load_state_dict(optim_param['optim_dict'])
        msg = f'resume optim path : {resume_optim_path}\n'
        logging.info(msg)

    warmup_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: min(1.0, epoch / args.WARMUP_ITER))
    training_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: (1 - epoch / args.EPOCH) ** 0.9)
    if args.resume is not None:
        for i in range(args.resume_epoch+1):
            training_scheduler.step()
            print(f"{i} epoch learning rate is : {optim.state_dict()['param_groups'][0]['lr']:.6f}")

    # view the learning rate in different layers
    # 打印每一层的名称及当前学习率
    # for name, param in clip_model.image_encoder.named_parameters():
    #     for param_group in optim.param_groups:
    #         try:
    #             if param in param_group['params']:
    #                 print(f"层名: {name}, 学习率: {param_group['lr']}")
    #                 break
    #         except:
    #             print('there has some wrong!!!!!!!!!!!!!')
    #
    # for name, param in clip_model.text_encoder.named_parameters():
    #     for param_group in optim.param_groups:
    #         try:
    #             if param in param_group['params']:
    #                 print(f"层名: {name}, 学习率: {param_group['lr']}")
    #                 break
    #         except:
    #             print('there has some wrong!!!!!!!!!!!!!')
    #
    # for name, param in clip_model.text_pooler.named_parameters():
    #     for param_group in optim.param_groups:
    #         try:
    #             if param in param_group['params']:
    #                 print(f"层名: {name}, 学习率: {param_group['lr']}")
    #                 break
    #         except:
    #             print('there has some wrong!!!!!!!!!!!!!')

    loss_record = []
    clip_model.image_encoder.train()
    start_epoch = 0

    # MAX_EPOCH = args.EPOCH
    if args.WARMUP:
        # MAX_EPOCH += 1
        start_epoch -= 1
    if args.resume is not None:
        start_epoch = args.resume_epoch + 1
    clip_model.train()
    clip_model.image_encoder.train()
    clip_model.text_encoder.train()

    forward_count = 0
    final_loss = {}
    MF_FREQ = args.MF_freq
    optim.zero_grad()
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
            sample_count = 0
            run_count = 0
            image_list, x_textVec = [], []
            load_finish = False
            while sample_count < 2000:
                try:
                    image, textVec, sample_info = next(train_data_iter)
                    for i in range(len(textVec)):
                        if textVec[i] in x_textVec:
                            run_count += 1
                            continue
                        else:
                            add_image = image[i, ...]
                            add_image = torch.unsqueeze(add_image, dim=0)
                            image_list.append(add_image)
                            x_textVec.append(textVec[i])
                            sample_count += 1
                        run_count += 1
                        if sample_count == args.batch_size:
                            x_image = torch.cat(image_list, dim=0)
                            load_finish = True
                            break
                except StopIteration:
                    train_data_iter = iter(train_dataloader)

                if load_finish:
                    break

            # x_img = sitk.GetImageFromArray(x_image.numpy()[0][0])
            # sitk.WriteImage(x_img, f'/sdf/yd/CLIP_Image_Project/grocery/x_{iteration}.nii.gz')
            # # y_img = sitk.GetImageFromArray(x_textVec.numpy()[0][0])
            # # sitk.WriteImage(y_img, f'/sdf/yd/CLIP_Image_Project/grocery/y_{iteration}')
            # continue

            # see_txt = f'/sdf/yd/CLIP_Image_Project/grocery/{iteration}_txt'
            # with open(see_txt, 'w') as f:
            #     for it in x_textVec:
            #         f.write(f'{it}\n')
            # continue

            x_image = x_image.to(args.device)
            # x_textVec = x_textVec.to(args.device)
            torch.autograd.set_detect_anomaly(True)
            if iteration == 0:
                optim.zero_grad()
                forward_count = 0
            input_data = {'x_img': x_image, 'text_data': x_textVec}
            if hasattr(args, 'eps'):
                input_data['eps'] = args.eps
            if args.Mix_Precision:
                with amp.autocast():
                    imageVector, textVec, logit_image, logit_text = clip_model(**input_data)
                    input_dict = {'image_features': imageVector, 'text_features': textVec,
                                  'logits_per_image': logit_image, 'logits_per_text': logit_text,
                                  'text_list': x_textVec, 'configs': args}
                    if hasattr(args, 'eps'):
                        input_dict['eps'] = args.eps
                    loss, loss_detail = clip_model.get_loss(input_dict, mode=args.loss_mode)
                    loss = loss / MF_FREQ
                    if (forward_count == 0):
                        final_loss['loss'] = loss.item()
                        for k in loss_detail.keys():
                            if isinstance(loss_detail[k], list):
                                final_loss[k] = [it / MF_FREQ for it in loss_detail[k]]
                            else:
                                final_loss[k] = loss_detail[k] / MF_FREQ
                    else:
                        final_loss['loss'] += loss.item()
                        for k in loss_detail.keys():
                            if isinstance(loss_detail[k], list):
                                for i in range(len(loss_detail[k])):
                                    final_loss[k][i] += loss_detail[k][i] / MF_FREQ
                            else:
                                final_loss[k] += loss_detail[k] / MF_FREQ
                forward_count += 1
                scaler.scale(loss).backward()
                if (forward_count % args.MF_freq == 0):
                    if hasattr(args, 'clip_grad') and args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=args.clip_grad_max)

                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    forward_count = 0
            else:
                imageVector, textVec, logit_image, logit_text = clip_model(x_image, x_textVec)
                input_dict = {'image_features': imageVector, 'text_features': textVec,
                              'logits_per_image': logit_image, 'logits_per_text': logit_text,
                              'text_list': x_textVec, 'configs': args}
                if hasattr(args, 'eps'):
                    input_dict['eps'] = args.eps
                loss, loss_detail = clip_model.get_loss(input_dict, mode=args.loss_mode)
                loss = loss / MF_FREQ
                if (forward_count == 0):
                    final_loss['loss'] = loss.item()
                    for k in loss_detail.keys():
                        if isinstance(loss_detail[k], list):
                            final_loss[k] = [it / MF_FREQ for it in loss_detail[k]]
                        else:
                            final_loss[k] = loss_detail[k] / MF_FREQ
                else:
                    final_loss['loss'] += loss.item()
                    for k in loss_detail.keys():
                        if isinstance(loss_detail[k], list):
                            for i in range(len(loss_detail[k])):
                                final_loss[k][i] += loss_detail[k][i] / MF_FREQ
                        else:
                            final_loss[k] += loss_detail[k] / MF_FREQ

                forward_count += 1
                loss.backward()
                if (forward_count % args.MF_freq == 0):
                    if hasattr(args, 'clip_grad') and args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=args.clip_grad_max)
                    optim.step()
                    optim.zero_grad()
                    forward_count = 0

            loss_record.append(loss.item())

            torch.cuda.synchronize()
            time_iter_end = time.time()

            if loss.item() > 100000000:
                print('model grad vanish, exit system!')
                sys.exit()
            else:
                pass

            loss_detail_text = ''
            if isinstance(final_loss, list):
                for it in final_loss:
                    loss_detail_text += f'_{it:.3f}'
            else:
                for k in final_loss.keys():
                    loss_detail_text += f'_{k}'
                    if isinstance(final_loss[k], list):
                        for it in final_loss[k]:
                            loss_detail_text += f'_{it:.3f}'
                    else:
                        loss_detail_text += f'_{final_loss[k]:.3f}'

            if forward_count % MF_FREQ == 0:
                mark = '***'
            else:
                mark = ''
            logging.info(f"{mark}epoch {epoch}: {iteration + 1}/{iter_nums}, loss:{final_loss['loss']:.3f}, "
                         f"detail:{loss_detail_text}, "
                         f"time:{(time_iter_end - time_iter_start): .3f}, "
                         f"image_lr: {optim.state_dict()['param_groups'][0]['lr']:.6f}, "
                         f"text_lr: {optim.state_dict()['param_groups'][1]['lr']:.6f}, "
                         f"run_count: {run_count}")

            if epoch == -1 and (((iteration + 1) % args.warm_up_val_space == 0) or (iteration == 250)):
                '''
                validation in warm up part
                '''
                clip_model.set_return_logit(False)
                logging.info('********* validation now *********')
                clip_model.eval()
                clip_model.image_encoder.eval()
                clip_model.text_encoder.eval()
                clip_model.text_pooler.eval()
                val_image_root = os.path.join(args.data_root, 'clip_brain_mri')
                val_save_dir = os.path.join(args.validation_save_root, args.model_name, f'E{epoch}', f'IT{iteration}')
                val_dataset = CLIP_Dataloader_Both_v2(names_root=train_names_path, image_root=val_image_root,
                                                      data_transforms=None, text_json_root=train_textVec_root,
                                                      image_flip_prob=0, return_val=True, dataset_tag='testing',
                                                      image_file_tag='nii.gz', json_version=args.json_version,
                                                      include_sample=args.include_sample,
                                                      replace_norm=args.replace_norm,
                                                      image_dir_append=args.image_from_dir,
                                                      dataset_version=args.dataset_version, configs=args)
                val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                                            pin_memory=False, drop_last=True)
                val_data_iter = iter(val_dataloader)
                if os.path.exists(val_save_dir) == False:
                    os.makedirs(val_save_dir)
                validation_Both_TopK(model=clip_model, text_include=val_dataset.text_include, dataiter=val_data_iter,
                                     save_root=val_save_dir, device=args.device, epoch=epoch)

                clip_model.image_encoder.train()
                clip_model.text_encoder.train()
                clip_model.text_pooler.train()
                clip_model.train()
                if hasattr(args, 'return_logit'):
                    clip_model.set_return_logit(args.return_logit)
                logging.info('********* validation end *********')

        time_epoch_end = time.time()
        logging.info(f"TRAIN EPOCH {epoch}, image_lr: {optim.state_dict()['param_groups'][0]['lr']:.6f}, "
                     f"text_lr: {optim.state_dict()['param_groups'][1]['lr']:.6f}, "
                     f"time: {(time_epoch_end - time_epoch_start):.3f}, ")

        plt_losses(loss_record, epoch)
        if args.WARMUP and epoch == -1:
            pass
        else:
            training_scheduler.step()

        if ((epoch + 1) % args.Checkpoint_Save_Space == 0):
            model_save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
            torch.save(clip_model, model_save_path)
            optim_save_path = os.path.join(snapshot_path, f'optim_{epoch}.pth')
            torch.save({'optim_dict': optim.state_dict()}, optim_save_path)

        '''
                validation part
                '''
        Validation_flag = False
        if epoch < 100:
            if (epoch == 0) or ((epoch + 1) % args.VALIDATION_SPACE == 0):
                Validation_flag = True
        else:
            if ((epoch + 1) % args.VALIDATION_SPACE == 0):
                Validation_flag = True
        if Validation_flag:
            clip_model.set_return_logit(False)
            logging.info('********* validation now *********')
            clip_model.eval()
            clip_model.image_encoder.eval()
            clip_model.text_encoder.eval()
            clip_model.text_pooler.eval()
            val_image_root = os.path.join(args.data_root, 'clip_brain_mri')
            val_save_dir = os.path.join(args.validation_save_root, args.model_name, f'E{epoch}')
            val_dataset = CLIP_Dataloader_Both_v2(names_root=train_names_path, image_root=val_image_root,
                                                  data_transforms=None, text_json_root=train_textVec_root,
                                                  image_flip_prob=0, return_val=True, dataset_tag='testing',
                                                  image_file_tag='nii.gz', json_version=args.json_version,
                                                  include_sample=args.include_sample, replace_norm=args.replace_norm,
                                                  image_dir_append=args.image_from_dir,
                                                  dataset_version=args.dataset_version, configs=args)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                                        pin_memory=False, drop_last=True)
            val_data_iter = iter(val_dataloader)
            if os.path.exists(val_save_dir) == False:
                os.makedirs(val_save_dir)
            validation_Both_TopK(model=clip_model, text_include=val_dataset.text_include, dataiter=val_data_iter,
                                 save_root=val_save_dir, device=args.device, epoch=epoch)

            clip_model.image_encoder.train()
            clip_model.text_encoder.train()
            clip_model.text_pooler.train()
            clip_model.train()
            if hasattr(args, 'return_logit'):
                clip_model.set_return_logit(args.return_logit)
            logging.info('********* validation end *********')

        if (epoch + 1) % 100 == 0:
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            handler = logger.handlers[0]  # 获取当前的文件处理器
            handler.close()  # 关闭当前文件处理器

            log_file = snapshot_path + f"/log_Start_E{epoch + 1}.txt"

            # 创建新的文件处理器
            new_handler = logging.FileHandler(log_file, mode='a')
            new_handler.setLevel(logging.INFO)
            new_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))

            # 替换当前的文件处理器
            logger.handlers[0] = new_handler

    model_save_path = os.path.join(snapshot_path, f'epoch_{args.EPOCH - 1}.pth')
    if not os.path.exists(model_save_path):
        torch.save(clip_model, model_save_path)
        optim_save_path = os.path.join(snapshot_path, f'optim_{args.EPOCH - 1}.pth')
        torch.save({'optim_dict': optim.state_dict()}, optim_save_path)


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
    parser.add_argument('--data_root', type=str, default=ProjectDir + f'/DATA/',
                        help='directory name which reserves the npy images')
    # parser.add_argument('--validation_save_root', '-validation_save_root', default=f'{ProjectDir}/Validation_View')
    parser.add_argument('--yaml_file_path', '-yaml_file_path',
                        default=f'{ProjectDir}/code/Train_yaml/{TrainingFileName}.yaml')
    parser.add_argument('--bert_root', '-bert_root', default=f"{ProjectDir}/Text_Model")
    parser.add_argument('--resume', '-resume', default=None)

    args = parser.parse_args()
    add_yaml_params(args, yaml_path=args.yaml_file_path)
    args.validation_save_root = f'{ProjectDir}/{args.ValidationNII_Dir}'
    args.bert_param_path = os.path.join(args.bert_root, args.bert_weight_name)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    snapshot_path = os.path.join(ProjectDir, args.Save_Dir, args.model_name)

    if args.resume is not None:
        assert args.resume_epoch is not None
        args.WARMUP = False
        # resume_path = os.path.join(snapshot_path, str(args.resume_epoch))
        # flag = input('if you want to resume training?\n'
        #              f'the resume_path is {snapshot_path}, the epoch is {args.resume_epoch}. y/n')
        # flag = flag.lower()
        # assert flag in ['y', 'n']
        # if flag != 'y':
        #     sys.exit()
    else:
        check_and_backup()
        pass

    if args.resume is not None:
        resume_model_path = os.path.join(snapshot_path, f'epoch_{args.resume_epoch}.pth')
        clip_model = torch.load(resume_model_path)
        clip_model = clip_model.to(args.device)
        clip_model.train()
        msg = f'resume model path : {resume_model_path}\n'
        # logging.info(msg)
    else:
        clip_frame = getattr(CLIP_model, args.CLIP_frame)
        clip_model = clip_frame(args)
        clip_model = clip_model.to(args.device)

    # log模块的设定
    if args.resume is not None:
        logging.basicConfig(filename=snapshot_path + f"/log_resume_start_{args.resume_epoch}.txt", level=logging.INFO,
                            filemode='a', format='[%(asctime)s] %(message)s')
        logging.info(msg)
    else:
        logging.basicConfig(filename=snapshot_path + f"/log.txt", level=logging.INFO,
                            filemode='a', format='[%(asctime)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(os.path.basename(os.path.abspath(__file__)))
    logger = logging.getLogger()
    logging.info(args.introduction)

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