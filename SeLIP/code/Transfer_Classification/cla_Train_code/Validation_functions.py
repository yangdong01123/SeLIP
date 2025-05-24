import logging
import os.path
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score



def validation_model_modal_embedding_ReturnAcc(model, val_dataloader, save_dir, epoch):
    results_list = []
    for x, label, sample in tqdm(val_dataloader):
        # 获取文件名并排序
        name = sample['name']
        mode = sample['mode']
        print(f'{name} is inferenceing')


        x = x.cuda()
        with torch.no_grad():
            output = model(x)
            output = F.softmax(output, dim=-1)
            output = output.cpu().numpy()
            pred_score = output[0][1]
            output = np.argmax(output, axis=1)[0]
        label = sample['label']
        results_list.append(
            {'name': name, 'mode': mode, 'label': label.item(), 'pred': output.item(), 'pred_score': pred_score.item()})

    try:
        df = pd.DataFrame(results_list)
        pred_list = df['pred'].tolist()
        pred_score_list = df['pred_score'].tolist()
        label_list = df['label'].tolist()
        # 计算准确率
        accuracy = accuracy_score(label_list, pred_list)
        logging.info(f"Accuracy: {accuracy:.4f}")

        # 计算精确率
        precision = precision_score(label_list, pred_list)
        logging.info(f"Precision: {precision:.4f}")

        # 计算召回率
        recall = recall_score(label_list, pred_list)
        logging.info(f"Recall: {recall:.4f}")

        # 计算 F1-score
        f1 = f1_score(label_list, pred_list)
        logging.info(f"F1-score: {f1:.4f}")

        auc = roc_auc_score(label_list, pred_score_list)
        logging.info(f"AUC-score: {auc:.4f}")

        # 计算混淆矩阵
        cm = confusion_matrix(label_list, pred_list)
        logging.info("Confusion Matrix:")
        logging.info(cm)

        save_path = os.path.join(save_dir, f'{epoch}.csv')
        df.to_csv(save_path)
        return accuracy
    except:
        return -1
