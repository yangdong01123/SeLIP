import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from transformers import BertModel, BertTokenizer
from CLIP_network.ResNet import generate_model


class BertPooler_D(nn.Module):
    def __init__(self, hidden_size, output_size, act=None):
        super().__init__()
        self.act = act
        self.dense = nn.Linear(hidden_size, output_size)
        if act:
            self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class CLIP_Both_model(nn.Module):
    def __init__(self, configs):
        super(CLIP_Both_model, self).__init__()
        self.device = configs.device
        self.truncation = configs.truncation
        self.padding = configs.padding
        self.max_length = configs.max_length
        self.return_tensors = configs.return_tensors
        self.bert_attention_mask = configs.bert_attention_mask
        self.return_logit = False
        if hasattr(configs, 'return_logit'):
            self.return_logit = configs.return_logit
        self.logit_sigmoid = False
        if hasattr(configs, 'logit_sigmoid'):
            self.logit_sigmoid = configs.logit_sigmoid

        self.image_encoder = generate_model(configs.image_model_depth, **configs.image_encoder_params)

        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_param_path)
        self.text_encoder = BertModel.from_pretrained(configs.bert_param_path)
        pooler_act = False
        if hasattr(configs, 'bert_pooler_act'):
            pooler_act = configs.bert_pooler_act
        self.text_pooler = BertPooler_D(768, configs.text_output_dim, act=pooler_act)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sigmoid = nn.Sigmoid()

        if configs.loss_mode == 'align_featureAlign':
            self.image_feature_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.text_feature_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def tokenizer_sentence(self, sentence_list):
        if self.bert_attention_mask:
            tokened_dict = self.tokenizer(sentence_list, truncation=self.truncation, padding=self.padding,
                                          max_length=self.max_length, return_tensors=self.return_tensors)
            return tokened_dict
        else:
            encoded_List = []
            for sen in sentence_list:
                sen_encoded = self.tokenizer.encode(sen, truncation=self.truncation, padding=self.padding,
                                                    max_length=self.max_length,
                                                    return_tensors=self.return_tensors)
                encoded_List.append(sen_encoded)

            out = torch.cat(encoded_List, dim=0)
            return out

    def generate_label(self, text):
        b = len(text)
        label = torch.eye(b)

        for i in range(b):
            t_i = text[i]
            for j in range(b):
                if i == j:
                    continue
                t_j = text[j]
                if (t_i == t_j):
                    label[i][j] = 1
                    label[j][i] = 1

        # for_see = label.numpy()
        label = Variable(label).to(self.device)
        return label

    def set_bert_attention_mask(self, mode):
        self.bert_attention_mask = mode

    def text_inference(self, text_data, eps=1e-8):
        if self.bert_attention_mask:
            text_token = self.tokenizer_sentence(text_data)
            text_features = self.text_encoder(input_ids=text_token['input_ids'].to(self.device),
                                              attention_mask=text_token['attention_mask'].to(self.device),
                                              token_type_ids=text_token['token_type_ids'].to(
                                                  self.device)).last_hidden_state
            text_features = self.text_pooler(text_features)

            if len(text_features.shape) == 3:
                text_features = text_features[:, 0, :]
            text_norm = text_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_norm.clamp(min=eps)
            return text_features
        else:
            text_token = self.tokenizer_sentence(text_data)
            text_features = self.text_encoder(text_token.to(self.device)).last_hidden_state
            text_features = self.text_pooler(text_features)

            if len(text_features.shape) == 3:
                text_features = text_features[:, 0, :]
            text_norm = text_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_norm.clamp(min=eps)
            return text_features

    def image_inference(self, x_image, eps=1e-3):
        image_features = self.image_encoder(x_image)
        image_norm = image_features.norm(dim=1, keepdim=True)
        image_features = image_features / image_norm.clamp(min=eps)
        return image_features


    def set_return_logit(self, mode):
        self.return_logit = mode


    def forward(self, x_img, text_data, modal=None, eps=1e-8):
        # x_img, text_features = x_pair['image'], x_pair['text']
        if modal is not None:
            image_features = self.image_encoder(x_img, modal)
        else:
            image_features = self.image_encoder(x_img)
        text_token = self.tokenizer_sentence(text_data)
        if self.bert_attention_mask:
            text_features = self.text_encoder(input_ids=text_token['input_ids'].to(self.device),
                                              attention_mask=text_token['attention_mask'].to(self.device),
                                              token_type_ids=text_token['token_type_ids'].to(
                                                  self.device)).last_hidden_state
        else:
            text_features = self.text_encoder(text_token.to(self.device)).last_hidden_state
        text_features = self.text_pooler(text_features)

        # normalized features
        image_norm = image_features.norm(dim=1, keepdim=True)
        image_features = image_features / image_norm.clamp(min=eps)

        if len(text_features.shape) == 3:
            text_features = text_features[:, 0, :]
        text_norm = text_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_norm.clamp(min=eps)

        if self.return_logit:
            image_features = torch.unsqueeze(image_features, dim=0)
            text_features = torch.unsqueeze(text_features, dim=0)

            # cosine similarity as logits
            self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.transpose(-1, -2)
            logits_per_image = logits_per_image[0]
            logits_per_text = logits_per_image.transpose(-1, -2)

            if self.logit_sigmoid:
                logits_per_image = self.sigmoid(logits_per_image)
                logits_per_text = self.sigmoid(logits_per_text)

            return image_features, text_features, logits_per_image, logits_per_text
        else:
            return image_features, text_features

    '''cal align dice part'''
    def cal_align_loss(self, image_features, text_features, labels=None):
        batch_size = image_features.shape[0]
        if labels is None:
            labels = Variable(torch.LongTensor(range(batch_size))).to(image_features.device)

        image_features = torch.unsqueeze(image_features, dim=0)
        text_features = torch.unsqueeze(text_features, dim=0)

        # cosine similarity as logits
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.transpose(-1, -2)
        logits_per_image = logits_per_image[0]
        logits_per_text = logits_per_image.transpose(-1, -2)

        align_loss0 = nn.CrossEntropyLoss()(logits_per_image, labels)
        align_loss1 = nn.CrossEntropyLoss()(logits_per_text, labels)

        return align_loss0, align_loss1

    def get_align_loss(self, image_features, text_features, labels=None, text_list=None, loss_alpha=None, loss_set=None,
                       word_vector=None, KL_mode=None, beta=None, configs=None):
        align_loss0, align_loss1 = self.cal_align_loss(image_features, text_features, labels=labels)
        loss = (align_loss0 + align_loss1) / 2
        return loss, [loss.item(), align_loss0.item(), align_loss1.item()]

    '''
    cal text soft label loss
    cal dice between texts, and use dice as the soft label
    '''
    def calculate_text_dice_coefficient(self, text1, text2):
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2)
        dice_coefficient = (2 * intersection) / union
        return dice_coefficient

    def cal_text_dice_matrix(self, text_list, device):
        C = len(text_list)
        matrix = torch.zeros((C, C), device=device)

        for i in range(C):
            t1 = text_list[i]
            for j in range(i, C):
                t2 = text_list[j]
                dc = self.calculate_text_dice_coefficient(t1, t2)
                matrix[i][j] = dc
                matrix[j][i] = dc
        return matrix

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def _soft_xent_loss_no_softmax(self, input, target):
        loss = -(target * input.log()).sum() / input.shape[0]
        return loss

    def cal_dice_soft_label_loss(self, image_features, text_features, text_list, labels=None):
        batch = image_features.shape[0]
        image_features = torch.unsqueeze(image_features, dim=0)
        text_features = torch.unsqueeze(text_features, dim=0)

        # cosine similarity as logits
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.transpose(-1, -2)
        logits_per_image = logits_per_image[0]
        logits_per_text = logits_per_image.transpose(-1, -2)

        text_dice_matrix = self.cal_text_dice_matrix(text_list, device=image_features.device)
        text_dice_softmax = F.softmax(text_dice_matrix, dim=1)

        text_dice_softmax += torch.eye(batch)

        loss_per_image = self._soft_xent_loss(logits_per_image, text_dice_softmax)
        loss_per_text = self._soft_xent_loss(logits_per_text, text_dice_softmax)
        loss = (loss_per_image + loss_per_text) / 2

        return loss, [loss.item(), loss_per_image.item(), loss_per_text.item()]

    '''cal soft KL Div in negative samples'''
    def remove_positive_samples(self, tensor_matrix):
        matrix_size = tensor_matrix.size(0)
        # 创建索引掩码
        mask = torch.eye(matrix_size, dtype=torch.bool)
        # 使用掩码获取新的矩阵
        new_matrix = tensor_matrix[~mask].view(matrix_size, matrix_size - 1)
        return new_matrix

    def cal_logits(self, image_features, text_features):
        batch = image_features.shape[0]
        image_features = torch.unsqueeze(image_features, dim=0)
        text_features = torch.unsqueeze(text_features, dim=0)

        # cosine similarity as logits
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.transpose(-1, -2)
        logits_per_image = logits_per_image[0]
        logits_per_text = logits_per_image.transpose(-1, -2)
        return logits_per_image, logits_per_text

    def cal_align_loss_by_logits(self, logits_per_image=None, logits_per_text=None, image_features=None,
                                            text_features=None, labels=None, text_list=None, configs=None, eps=1e-8):
        batch_size = logits_per_image.shape[0]
        labels = Variable(torch.LongTensor(range(batch_size))).to(logits_per_image.device)
        align_loss0 = nn.CrossEntropyLoss()(logits_per_image, labels)
        align_loss1 = nn.CrossEntropyLoss()(logits_per_text, labels)

        align_loss = (align_loss0 + align_loss1) / 2
        return align_loss, {'align': align_loss.item(), 'align_det': [align_loss0.item(), align_loss1.item()]}

    def get_feature_cosine_sim_matrix(self, features, device):
        features_norm = features.norm(dim=1, keepdim=True)
        features = features / features_norm

        sim = features @ features.T
        return sim

    def cal_single_line_text_sim(self, text1, text2, display_tags=("高信号", "低信号", "长信号", "短信号", "等信号"), weight=(0.5, 0.5)):
        t1_split = text1.split("表现为")
        t1_split[0] = t1_split[0].replace('在', '')
        t2_split = text2.split("表现为")
        t2_split[0] = t2_split[0].replace('在', '')

        finded = [0, 0]
        for tag in display_tags:
            if finded[0] == 0 and tag in t1_split[1]:
                t1_split.append(tag)
                finded[0] = 1
            if finded[1] == 0 and tag in t2_split[1]:
                t2_split.append(tag)
                finded[1] = 1
            if finded[0] == 1 and finded[1] == 1:
                break

        if finded[0] == 0:
            t1_split.append(t1_split[1])
        if finded[1] == 0:
            t2_split.append(t2_split[1])

        score = 0
        if t1_split[0] == t2_split[0]:
            score += weight[0]
        if t1_split[2] == t2_split[2]:
            score += weight[1]

        dice = self.calculate_text_dice_coefficient(t1_split[0] + t1_split[1], t2_split[0] + t2_split[1])
        return score, dice

    def cal_text_sim_score(self, text1, text2, weight=(0.5, 0.5), have_modal=False):
        assert weight[0] + weight[1] == 1
        if have_modal:
            t1_list = text1.split("。")
            t1_list = [item.split("，")[1] for item in t1_list]
            t2_list = text2.split("。")
            t2_list = [item.split("，")[1] for item in t2_list]
        else:
            t1_list = text1.split("，")
            t2_list = text2.split("，")

        score1 = 0
        for t1 in t1_list:
            temp = 0
            for t2 in t2_list:
                sc, dc = self.cal_single_line_text_sim(t1, t2, weight=weight)
                temp += sc * dc
            score1 += temp / t2_list.__len__()
        score1 = score1 / t1_list.__len__()

        score = min(score1, 1)
        return score




    def cal_text_sim_matrix(self, text_list, device='cuda', weight=(0.5, 0.5),
                            healthy="头颅形态、大小未见异常。脑实质未见异常信号影，所见脑室形态及脑沟裂池未见异常扩张或变窄，中线结构未见移位。",
                            have_modal=False):
        C = len(text_list)
        matrix = torch.zeros((C, C), device=device)

        for i in range(C):
            t1 = text_list[i]
            for j in range(i, C):
                t2 = text_list[j]
                if (i == j):
                    matrix[i][j] = 1
                    continue
                if(t1 == healthy or t2 == healthy):
                    matrix[i][j] = 0
                    matrix[j][i] = 0
                else:
                    sim = self.cal_text_sim_score(t1, t2, weight=weight, have_modal=have_modal)
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        return matrix

    def cal_align_TextSimNegKL(self, logits_per_image, logits_per_text, image_features=None,
                                            text_features=None, labels=None, text_list=None, configs=None, eps=1e-8):
        align_loss, align_detail = self.cal_align_loss_by_logits(logits_per_image, logits_per_text)
        kl_loss, kl_detail = self.cal_TextSimNegKLDiv(logits_per_image=logits_per_image,
                                                      logits_per_text=logits_per_text, configs=configs,
                                                      text_list=text_list, eps=eps)
        loss = align_loss * configs.loss_alpha['align'] + kl_loss * configs.loss_alpha['NegKL']
        return loss, {**align_detail, **kl_detail}


    def cal_TextSimNegKLDiv(self, image_features=None, text_features=None, logits_per_image=None,
                                logits_per_text=None, text_list=None, labels=None, eps=1e-8, configs=None):
        assert (image_features is not None and text_features is not None) or (logits_per_image is not None and logits_per_text is not None)
        have_modal = False
        if hasattr(configs, 'have_modal'):
            have_modal = configs.have_modal
        if hasattr(configs, 'text_sim_weight'):
            sim_matrix = self.cal_text_sim_matrix(text_list, configs.device, weight=configs['text_sim_weight'], have_modal=have_modal)
        else:
            sim_matrix = self.cal_text_sim_matrix(text_list, configs.device, have_modal=have_modal)

        if configs.loss_set['include_positive']:
            logits_per_image_neg = logits_per_image
            logits_per_text_neg = logits_per_text
            sim_matrix_neg = sim_matrix
        else:
            logits_per_image_neg = self.remove_positive_samples(logits_per_image)
            logits_per_text_neg = self.remove_positive_samples(logits_per_text)
            sim_matrix_neg = self.remove_positive_samples(sim_matrix)

        logits_per_image_neg = F.softmax(logits_per_image_neg, dim=1)
        logits_per_text_neg = F.softmax(logits_per_text_neg, dim=1)

        flag = False
        if hasattr(configs, 'loss_set'):
            if 'sim_norm_mode' in configs.loss_set.keys():
                flag = True
                if configs.loss_set['sim_norm_mode'] == 'linear':
                    sim_sum = torch.sum(sim_matrix_neg, dim=-1, keepdim=True)
                    sim_sum = torch.clamp(sim_sum, min=eps)
                    sim_matrix_neg = sim_matrix_neg / sim_sum
                else:
                    sim_matrix_neg = F.softmax(sim_matrix_neg)

        if not flag:
            sim_matrix_neg = F.softmax(sim_matrix_neg)

        sim_matrix_neg = torch.clamp(sim_matrix_neg, eps, 1)
        logits_per_image_neg = torch.clamp(logits_per_image_neg, eps, 1)
        logits_per_text_neg = torch.clamp(logits_per_text_neg, eps, 1)

        assert configs.loss_set['KL_way'] == 1 or configs.loss_set['KL_way'] == 2
        if configs.loss_set['KL_way'] == 1:
            kl_image = F.kl_div(logits_per_image_neg.log(), sim_matrix_neg, reduction=configs.loss_set['KL_mode'])
            kl_text = F.kl_div(logits_per_text_neg.log(), sim_matrix_neg, reduction=configs.loss_set['KL_mode'])
            kl_loss = (kl_image + kl_text) / 2
            return kl_loss, {'KL': kl_loss.item(), 'KLIm': kl_image.item(),
                             'KLTe': kl_text.item()}
        else:
            kl_image_1 = F.kl_div(logits_per_image_neg.log(), sim_matrix_neg, reduction=configs.loss_set['KL_mode'])
            kl_image_2 = F.kl_div(sim_matrix_neg.log(), logits_per_image_neg, reduction=configs.loss_set['KL_mode'])
            kl_image = (kl_image_1 + kl_image_2) / 2

            kl_text_1 = F.kl_div(logits_per_text_neg.log(), sim_matrix_neg, reduction=configs.loss_set['KL_mode'])
            kl_text_2 = F.kl_div(sim_matrix_neg.log(), logits_per_text_neg, reduction=configs.loss_set['KL_mode'])
            kl_text = (kl_text_1 + kl_text_2) / 2

            kl_loss = (kl_image + kl_text) / 2
            return kl_loss, {'KL': kl_loss.item(), 'KLIm': [kl_image_1.item(), kl_image_2.item(), kl_image.item()],
                             'KLTe': [kl_text_1.item(), kl_text_2.item(), kl_text.item()]}

    def get_loss(self, input_data, mode):
        assert mode in ['align', 'align_TextSimNegKL']
        if mode == 'align':
            loss, loss_detail = self.cal_align_loss_by_logits(**input_data)
            return loss, loss_detail
        else :
            loss, loss_detail = self.cal_align_TextSimNegKL(**input_data)
            return loss, loss_detail


if __name__=="__main__":
    pass
