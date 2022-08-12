"""
Created by FENG YUAN on 2022/1/7
"""

import config
from model1 import Model
from dataset import CustomDataset, SqlLabelEncoder
from utils import read_data, read_tables

import os
import json
import datetime
from shutil import copyfile
from time import time

import numpy as np
import torch
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
import transformers


os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES  # specify GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
transformers.logging.set_verbosity_error()

def transform2sql(label_COND_CONN_OP, label_SEL_AGG, label_COND_OP, header_masks, label_encoder):
    '''
    label_COND_CONN_OP: shape (batch_size,)
    label_SEL_AGG: shape (batch_size, max_headers)
    label_COND_OP: shape (batch_size, max_headers)
    header_masks: shape (batch_size, max_headers)
    '''
    sqls = []
    for cond_conn_op, sel_agg, cond_op, header_masks in zip(label_COND_CONN_OP, label_SEL_AGG, label_COND_OP, header_masks):
        '''
        cond_conn_op: a single value
        sel_agg: shape (max_headers,)
        cond_op: shape (max_headers,)
        header_masks: shape (max_headers,)
        '''
        header_len = torch.sum(header_masks)  # a single value

        sel_agg = sel_agg[:header_len]
        cond_op = cond_op[:header_len]
        
        sql = label_encoder.decode(cond_conn_op, sel_agg, cond_op)
        sql['conds'] = [cond for cond in sql['conds'] if cond[0] < header_len]
        
        sel = []
        agg = []
        for col_id, agg_op in zip(sql['sel'], sql['agg']):
            if col_id < header_len:
                sel.append(col_id)
                agg.append(agg_op)
                
        sql['sel'] = sel
        sql['agg'] = agg
        sqls.append(sql)

    return sqls

def get_labels(logits):
    logits_softmax = torch.softmax(logits, dim=-1)      # (batch, max_len, num_classes)
    y_pred_tags = torch.argmax(logits_softmax, dim=-1)  # (batch, max_len)
    return y_pred_tags                 

def main():
    # Mode selection
    mode = input('Choose a mode (1 for evaluation, 2 for only inference):')
    mode = int(mode)

    # Checkpoint selection
    file_dict = dict()
    for i, f in enumerate(os.listdir(config.ROOT_MODEL_PATH)):
        if '.pt' in f:
            file_dict[i] = f
    print(file_dict)
    get_ckpt = input('Choose a checkpoint:')
    get_ckpt = int(get_ckpt)

    # Load data
    print('Loading data...')
    test_tables = read_tables(config.TEST_TABLE_FILE)
    test_data = read_data(config.TEST_DATA_FILE, test_tables)
    print(f'{len(test_data)} samples and {len(test_tables)} tables in the test set')
    print('Loading finished.')

    # Create dataset and dataloader
    print('Creating dataset and dataloader...')
    if mode == 1:     # not implemented
        test_set = CustomDataset(test_data)
    if mode == 2:
        test_set = CustomDataset(test_data, sql_label_encoder=None)
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print('Creating finished.')

    # Init model
    print('Initializing model...')
    model = Model()
    print(model)
    checkpoint = torch.load(config.ROOT_MODEL_PATH + file_dict[get_ckpt], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if config.ONLY_CPU:
        print('Using CPU for inference.')
    else:
        print('Using GPU for inference.')
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    print('Initializing finished.')

    # Inference
    t1 = time()

    sqls = []

    with torch.no_grad():
        model.eval()
        for _, batch in enumerate(test_dataloader):
            if config.ONLY_CPU:
                b_token_ids       = batch['token_ids']
                b_attention_masks = batch['attention_masks']
                b_header_ids      = batch['header_ids']
            else:
                b_token_ids       = batch['token_ids'].cuda()
                b_attention_masks = batch['attention_masks'].cuda()
                b_header_ids      = batch['header_ids'].cuda()

            output = model(
                token_ids=b_token_ids,
                attention_masks=b_attention_masks,
                header_ids=b_header_ids
            )

            # logits for each subtask
            logits_COND_CONN_OP = output['COND_CONN_OP']  # shape (batch_size, COND_CONN_OP classes)
            logits_SEL_AGG      = output['SEL_AGG']       # shape (batch_size, max_headers, SEL_AGG classes)
            logits_COND_OP      = output['COND_OP']       # shape (batch_size, max_headers, COND_OP classes)

            # will be used to transform to sql results
            b_header_masks = batch['header_masks']        # shape (batch_size, max_headers)

            # print(logits_COND_CONN_OP.shape)

            if config.ONLY_CPU:
                label_COND_CONN_OP = get_labels(logits_COND_CONN_OP).numpy()   # shape (batch_size, )
                label_SEL_AGG      = get_labels(logits_SEL_AGG).numpy()        # shape (batch_size, max_headers)
                label_COND_OP      = get_labels(logits_COND_OP).numpy()        # shape (batch_size, max_headers)
            else:
                label_COND_CONN_OP = get_labels(logits_COND_CONN_OP).cpu().numpy()
                label_SEL_AGG      = get_labels(logits_SEL_AGG).cpu().numpy()
                label_COND_OP      = get_labels(logits_COND_OP).cpu().numpy()

            # print(label_COND_CONN_OP) 
            # print(label_SEL_AGG[0])       
            # print(label_COND_OP[0])      

            ## transform to sql batch by batch
            b_sqls = transform2sql(label_COND_CONN_OP, label_SEL_AGG, label_COND_OP, b_header_masks, SqlLabelEncoder())
            # print(b_sqls)
            print(len(b_sqls)) 
            sqls.extend(b_sqls)

    t2 = time()

    print(f'Inference finished in {(t2 - t1):.2f}s, '
          f'{(t2 - t1) / len(test_dataloader):.4f}s for each batch, '
          f'{(t2 - t1) / len(test_data):.4f}s for each sample.')

    # Write to local
    with open(config.TASK1_RESULT, 'w') as f:
        for sql in sqls:
            json_str = json.dumps(sql, ensure_ascii=False)
            f.write(json_str + '\n')

    # print(local_label_COND_CONN_OP.shape)
    # print(local_label_SEL_AGG.shape)
    # print(local_label_COND_OP.shape)


if __name__ == "__main__":
    # execute only if run as a script
    main()
