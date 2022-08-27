import config
from dataset import CustomDataset

import os
import sys
import datetime
from time import time 
from shutil import copyfile


import joblib
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification
from focal_loss import FocalLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_labels(logits):
    y_pred_softmax = torch.nn.functional.softmax(logits, dim = -1)
    y_pred_probs, y_pred_tags = torch.max(y_pred_softmax, dim = -1)  
    return y_pred_tags, y_pred_probs

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc



def main():

    ## Loading and processing data
    print('Loading and processing test set...')
    test_df = pd.read_csv(config.TEST_FILE, encoding='utf-8-sig', sep='\t')
    le = joblib.load(config.LABEL_ENCODER_PATH)

    test_set = CustomDataset(
        sentences=test_df[config.CONTENT_FIELD].values.astype("str"),
        labels=None
    )

    test_dataloader = DataLoader(
        dataset=test_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    count_sample = test_df.shape[0]
    count_batch = len(test_dataloader)
    print(f'Load {count_sample} samples in {count_batch} batches, batch size {config.BATCH_SIZE}.') 
    print('Loading finished.')

    ## Init model
    print('Initializing model...')
    model = AlbertForSequenceClassification.from_pretrained(
        config.BASE_MODEL_PATH, 
        num_labels=len(le.classes_),
        output_attentions=False, 
        output_hidden_states=False
    )
    # print(model)
    checkpoint = torch.load(config.CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('    Using GPU for inference.')
    model = torch.nn.DataParallel(model)   
    model = model.cuda()
    print('Initialization finished.')

    ## Inference 
    t1 = time()

    local_labels = np.array([])
    local_probs = np.array([])

    with torch.no_grad():
        model.eval()
        for _, batch in enumerate(test_dataloader):

            b_token_ids = batch['token_ids'].cuda()
            b_attention_masks = batch['attention_masks'].cuda()
            b_token_type_ids = batch['token_type_ids'].cuda()
            
            output = model(
                input_ids=b_token_ids, 
                token_type_ids=b_token_type_ids, 
                attention_mask=b_attention_masks
            )

            logits = torch.squeeze(output.logits)

            # get labels
            test_labels, test_probs = get_labels(logits)
            test_labels = test_labels.cpu().numpy()
            test_probs = test_probs.cpu().numpy()

            local_labels = np.append(local_labels, test_labels)
            local_probs = np.append(local_probs, test_probs)

    
    t2 = time()
    print(f'Inference finished in {(t2-t1):.2f}s, {(t2-t1)/count_batch:.4f}s for each batch, {(t2-t1)/count_sample:.4f}s for each sample.')

    test_df['pred'] = local_labels
    test_df['pred'] = test_df['pred'].apply(lambda x: int(x))
    test_df['pred'] = le.inverse_transform(test_df['pred'])
    test_df['prob'] = local_probs

    test_df.to_csv(config.OUTPUT_TEST, index=False, encoding='utf-8-sig', sep='\t')
    
    print('All done!')

if __name__ == "__main__":

    main()

