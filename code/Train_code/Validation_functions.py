import pandas as pd
import torch
import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def validation_Both_TopK(model, text_include, dataiter, save_root, device, topk=10, MGPU=False, epoch=None):
    df = pd.DataFrame(text_include)
    df.to_excel(os.path.join(save_root, 'text_index.xlsx'))
    prepared_text = []
    for i in range(len(text_include)):
        text = [text_include[i][1]]
        with torch.no_grad():
            if MGPU:
                text_tk = model.tokenizer_sentence(text)
                t_vec = model.text_inference(text_tk)
            else:
                t_vec = model.text_inference(text)
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
    while True:
        try:
            x_image, x_text, info = next(dataiter)
        except StopIteration:
            break
        modal = info['modal'][0]
        name = info['name'][0]

        x_image = x_image.to(device)
        if 'modal_emb' in info.keys():
            modal_emb = info['modal_emb']
            modal_emb = torch.tensor(modal_emb, device=device)
            input_data = {'x_img': x_image, 'text_data': x_text, 'modal': modal_emb}
        else:
            input_data = {'x_img': x_image, 'text_data': x_text}

        with torch.no_grad():
            # if MGPU:
            #     text_tk = model.tokenizer_sentence(x_text)
            #     image_output, text_output = model(x_image, text_tk)
            # else:
            image_output, text_output = model(**input_data)
        # image_output = image_output.cpu()
        # text_output = text_output.cpu()

        save_dir = os.path.join(save_root, f'{name}_{modal}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logging.info(f'-----------------------------{name}_{modal}_start---------------------------------')
        image_save_path = os.path.join(save_dir, f"{modal}_imageVec.pt")
        torch.save(image_output, image_save_path)
        text_save_path = os.path.join(save_dir, f"{modal}_textVec.pt")
        torch.save(text_output, text_save_path)
        sim = torch.cosine_similarity(image_output, text_output)
        if epoch is not None:
            logging.info(f"E{epoch}: {name}_{modal}_Cosine_Similarity: {sim.item()}")
        else:
            logging.info(f"{name}_{modal}_Cosine_Similarity: {sim.item()}")
        sim_list.append(sim.item())

        sim_log = []
        for i in range(len(text_include)):
            sim = torch.cosine_similarity(image_output, prepared_text[i])
            # logging.info(f'{count}_{i}_sim: {sim.item()}')
            sim_log.append({'cosine_sim': sim.item(), 'text': text_include[i][1]})

        sorted_sim_log = sorted(sim_log, key=lambda x: x['cosine_sim'], reverse=True)
        log_df = pd.DataFrame(sorted_sim_log)
        save_path = os.path.join(save_dir, 'sim_log.xlsx')
        log_df.to_excel(save_path)
        txt_path = os.path.join(save_dir, f'TopK_record.txt')

        with open(txt_path, 'w') as f:
            f.write(f'{x_text[0]}\n')
            if epoch is not None:
                logging.info(f'E{epoch}: {x_text[0]}')
            else:
                logging.info(f'{x_text[0]}')
            f.write(f'-------------Top_{topk}---------------------:\n')
            for tp in range(topk):
                f.write(f"{sorted_sim_log[tp]['cosine_sim']}   {sorted_sim_log[tp]['text']}\n")
                if epoch is not None:
                    logging.info(f"E{epoch}: Top_{topk}_{tp} is : {sorted_sim_log[tp]['cosine_sim']}   {sorted_sim_log[tp]['text']}")
                else:
                    logging.info(
                        f"Top_{topk}_{tp} is : {sorted_sim_log[tp]['cosine_sim']}   {sorted_sim_log[tp]['text']}")

        error_count['count'] += 1
        find_flag = False
        for tp in range(10):
            if find_flag:
                error_count[f'top{tp + 1}'] += 1
                continue
            # a = sorted_sim_log[tp]['text']
            # a1 = x_text
            if x_text[0] == sorted_sim_log[tp]['text']:
                error_count[f'top{tp + 1}'] += 1
                find_flag = True
        logging.info(f'-----------------------------{name}_{modal}_end---------------------------------')
        count += 1

    logging.info('--------------------------------------------------------------')

    for i in range(1, 11):
        error_rate = error_count[f'top{i}'] / error_count['count']
        if epoch is not None:
            logging.info(f'E{epoch}: Top_{i}_right_rate: {error_rate}\n')
        else:
            logging.info(f'Top_{i}_right_rate: {error_rate}\n')

    sim_list = np.array(sim_list)
    mean = sim_list.mean()
    if epoch is not None:
        logging.info(f'E{epoch}: mean similarity is : {mean}')
    else:
        logging.info(f'mean similarity is : {mean}')
    logging.info('--------------------------------------------------------------')


if __name__=="__main__":
    pass

