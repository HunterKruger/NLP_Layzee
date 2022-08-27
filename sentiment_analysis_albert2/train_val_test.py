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
from dataset import split_df, CustomDataset

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

    ## Loading data
    print('Loading dataset...')
    df = pd.read_csv(config.INPUT_FILE, encoding='utf-8-sig', error_bad_lines=False, sep='\t').sample(50000)
    le = LabelEncoder()
    df[config.LABEL_FIELD] = le.fit_transform(df[config.LABEL_FIELD])
    joblib.dump(le, config.LABEL_ENCODER_PATH)

    train_df, val_df, test_df = split_df(
        df, 
        test_ratio=config.TEST_SIZE, 
        val_ratio=config.VALIDATION_SIZE, 
        target=None, 
        random_state=config.RANDOM_STATE
    )
    print('Training set shape: '+ str(train_df.shape))
    print('Validaiton set shape: '+ str(val_df.shape))
    print('Test set shape: ' + str(test_df.shape))
    print('Loading finished.')
    print('Saving training set & validation set & test set to local...')
    train_df.to_csv(config.TRAIN_FILE, index=False, sep='\t')
    val_df.to_csv(config.VALIDATION_FILE, index=False, sep='\t')
    test_df.to_csv(config.TEST_FILE, index=False, sep='\t')
    print('Saving finished.')
    
    ## Processing data
    print('Processing dataset...')
    train_set = CustomDataset(
        sentences=train_df[config.CONTENT_FIELD].values.astype("str"),
        labels=train_df[config.LABEL_FIELD]
    )
    val_set = CustomDataset(
        sentences=val_df[config.CONTENT_FIELD].values.astype("str"),
        labels=val_df[config.LABEL_FIELD]
    )
    train_dataloader = DataLoader(
        dataset=train_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        dataset=val_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print('Processing finished.')


    ## Model init
    print('Initializing model...')
    model = AlbertForSequenceClassification.from_pretrained(
        config.BASE_MODEL_PATH, 
        num_labels=len(le.classes_),
        output_attentions=False, 
        output_hidden_states=False
    )   # model with downstream by huggingface
    model = torch.nn.DataParallel(model.cuda())   # multi-gpu
    # print(model)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = params - trainable
    print(f'Parameters: {params}')
    print(f'Trainable: {trainable}')
    print(f'Untrainable: {untrainable}')
    print('Initialization finished.')

   
    ## Training and validation
    print('Start training...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.FACTOR, patience=config.REDUCE_PAT)
    # criterion = torch.nn.CrossEntropyLoss() 
    criterion = FocalLoss(gamma=2) 
    n_total_steps = len(train_dataloader)  

    # Early stopping parameters
    trigger_times = 0
    last_loss = 100
    best_model = None
    best_epoch = None

    for epoch in range(1, config.EPOCHS+1):   # iterate epochs
        
        t1 = time()

        # TRAINING
        train_epoch_loss, train_epoch_acc = 0, 0
        model.train()

        print('Current LR: ', optimizer.state_dict()['param_groups'][0]['lr'])

        for step, batch in enumerate(train_dataloader):  # iterate steps
            
            t_a = time()
            b_token_ids = batch['token_ids'].cuda()
            b_attention_masks = batch['attention_masks'].cuda()
            b_token_type_ids = batch['token_type_ids'].cuda()
            b_labels = batch['labels'].cuda()

            # forward-propagation
            output = model( 
                input_ids=b_token_ids, 
                token_type_ids=b_token_type_ids, 
                attention_mask=b_attention_masks,
                labels=b_labels
            )
            logits = output[1]       
            
            train_loss = criterion(logits, b_labels)  # loss for current step
            train_acc = multi_acc(logits, b_labels)   # accuracy for current step
                
            optimizer.zero_grad()                     # clear gradients
            train_loss.backward()                     # back-propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_NORM)   # clip gradients
            optimizer.step()                          # update parameters

            t_b = time()

            if (step+1) % config.LOG_INTERVAL == 0:        # print current step training loss and accuracy for every LOG_INTERVAL steps
               print(f'epoch {epoch}/{config.EPOCHS}, step {step+1}/{n_total_steps}, loss {train_loss.item():.4f}, acc {train_acc:.4f}, time {(t_b-t_a):.4f}s')

            train_epoch_loss += train_loss.item()          # sum training loss for current epoch
            train_epoch_acc += train_acc.item()            # sum training accuracy for current epoch

        # VALIDATION    
        with torch.no_grad():

            val_epoch_loss, val_epoch_acc = 0, 0
            model.eval()

            for step, batch in enumerate(val_dataloader):

                b_token_ids = batch['token_ids'].cuda()
                b_attention_masks = batch['attention_masks'].cuda()
                b_token_type_ids = batch['token_type_ids'].cuda()
                b_labels = batch['labels'].cuda()

                output = model(
                    input_ids=b_token_ids, 
                    token_type_ids=b_token_type_ids, 
                    attention_mask=b_attention_masks,
                    labels=b_labels
                )

                logits = output[1]

                val_loss = criterion(logits, b_labels)
                val_acc = multi_acc(logits, b_labels)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        t2 = time()
        
        print(f'Epoch {epoch}: | Train Loss: {train_epoch_loss/len(train_dataloader):.4f} | Val Loss: {val_epoch_loss/len(val_dataloader):.4f} | Train Acc: {train_epoch_acc/len(train_dataloader):.4f} | Val Acc: {val_epoch_acc/len(val_dataloader):.4f} | Time: {(t2-t1):.2f}s.')
        
        # Reduce LR on plateau
        scheduler.step(val_epoch_loss/len(val_dataloader))

        # Early stopping
        current_loss = val_epoch_loss/len(val_dataloader)
        print(f'    Current epoch loss: {current_loss:.4f}')
        print(f'    Last epoch loss: {last_loss:.4f}')

        if epoch == config.EPOCHS:
            print('No early stopping, saving model...')
            torch.save(
                {'model_state_dict': model.module.state_dict()},      # save parameters
                config.CKPT_PATH                                       # set path
            )
        if current_loss > last_loss:
            trigger_times += 1
            print('    Trigger times:', trigger_times)
            if trigger_times >= config.PATIENCE:
                print(f'Early stopping, saving the best model in epoch {best_epoch}!')
                torch.save(
                    {'model_state_dict': model.module.state_dict()},      # save parameters
                    config.CKPT_PATH                                       # set path
                )
                break
        else:
            print('    Trigger times: 0')
            trigger_times = 0
            best_model = model
            best_epoch = epoch
        last_loss = current_loss

    print('Training finished!')



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
    
    print('Saving config...')
    copyfile('config.py', config.CONFIG_FILE)         # save config
    print('Saving finished.')
    
    print('All done!')



if __name__ == "__main__":

    main()

