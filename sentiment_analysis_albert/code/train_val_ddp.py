import config
from model import Albert
from dataset import CustomDataset, process_data, train_test_split

import os
from time import time 
import datetime
import pandas as pd
from shutil import copyfile
import torch
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification
from transformers import get_polynomial_decay_schedule_with_warmup

### DDP
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()
args.is_master = (args.local_rank == 0)
dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
###


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


def main():
    
    ## Loading data
    if args.is_master:                     
        print('Loading dataset...')

    if config.ALREADY_SPLIT:
        train_df = pd.read_csv(config.TRAIN_FILE) 
        val_df = pd.read_csv(config.VALIDATION_FILE)  
        if args.is_master:                     
            print('Training set shape: '+ str(train_df.shape))
            print('Validaiton set shape: '+ str(val_df.shape))
            print('Loading finished.')
    else:
        data_df = process_data(config.INPUT_FILE, config.CLS2IDX, True)     # DataFrame, only used labeled data
        train_df, test_df = train_test_split(data_df, test_size=config.TEST_SIZE, shuffle=True, random_state=config.RANDOM_STATE)
        train_df, val_df = train_test_split(train_df, test_size=config.VALIDATION_SIZE, shuffle=True, random_state=config.RANDOM_STATE)  
        if args.is_master:                     
            print('Training set shape: '+ str(train_df.shape))
            print('Validaiton set shape: '+ str(val_df.shape))
            print('Test set shape: '+ str(test_df.shape))
            print('Loading finished.')
            print('Saving training set & validation set & test set to local...')
        train_df.to_csv(config.TRAIN_FILE, index=False)
        val_df.to_csv(config.VALIDATION_FILE, index=False)
        test_df.to_csv(config.TEST_FILE, index=False)
        if args.is_master:                     
            print('Saving finished.')
    

    ## Processing data
    if args.is_master:                     
        print('Processing dataset...')

    train_set = CustomDataset(sentences=train_df[config.CONTENT_FIELD].values.astype("str"), labels=train_df[config.LABEL_FIELD])
    val_set = CustomDataset(sentences=val_df[config.CONTENT_FIELD].values.astype("str"), labels=val_df[config.LABEL_FIELD])

    train_sampler = DistributedSampler(train_set)
    val_sampler = DistributedSampler(val_set)

    train_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=train_sampler)
    val_dataloader = DataLoader(val_set, batch_size=config.BATCH_SIZE, sampler=val_sampler)
    
    if args.is_master:                     
        print('Processing finished.')

    ## Modeling 
    model = AlbertForSequenceClassification.from_pretrained(
        config.BASE_MODEL_PATH, 
        num_labels=config.CLASSES,
        output_attentions=False, 
        output_hidden_states=False
    )    

    if args.is_master:                     
        print('Initializing model...')
        filenames = dict()
        i = 1
        for f in os.listdir(config.ROOT_PATH):
            if '.pt' in f:
                filenames[i] = f
                i+=1
        print(filenames)
        get_epoch = input('Choose a checkpoint (input 0 to train a new model):')
        get_epoch = int(get_epoch)
        if get_epoch!=0:
            checkpoint = torch.load(config.ROOT_PATH+filenames[get_epoch])
            model.load_state_dict(checkpoint['model_state_dict'])


    ### DDP
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    ###

    if args.is_master:
        print(model)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        untrainable = params - trainable
        print(f'Parameters: {params}')
        print(f'Trainable: {trainable}')
        print(f'Untrainable: {untrainable}')
        print('Initialization finished.')
        print('Start training...')


    ## Training and validation

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=config.LR, 
    #     momentum=config.MOMENTUM, 
    #     nesterov=config.NESTEROV, 
    #     weight_decay=config.WEIGHT_DECAY
    # )

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.WARMUP_STEPS, 
        num_training_steps=len(train_dataloader) * config.EPOCHS,
        lr_end=config.LR_END,
        power=config.POWER
    )   
    
    # loss
    criterion = torch.nn.CrossEntropyLoss() 

    # log recording
    history = {'train_loss': [], "val_loss": [], 'train_acc': [], "val_acc": []}

    n_total_steps = len(train_dataloader)  


    for epoch in range(1, config.EPOCHS + 1):   # iterate epochs
        
        t1 = time()

        # TRAINING
        train_epoch_loss, train_epoch_acc = 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):  # iterate steps
            
            t_a = time()
            b_token_ids = batch['token_ids'].cuda(non_blocking=True)
            b_attention_masks = batch['attention_masks'].cuda(non_blocking=True)
            b_token_type_ids = batch['token_type_ids'].cuda(non_blocking=True)
            b_labels = batch['labels'].cuda(non_blocking=True)

            # forward-propagation
            output = model(input_ids=b_token_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_masks, labels=b_labels)

            logits = output[1]

            train_loss = criterion(logits, b_labels)  # loss for current step
            train_acc = multi_acc(logits, b_labels)   # accuracy for current step
                            
            optimizer.zero_grad()                    # clear gradients
            train_loss.backward()                    # back-propagation
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config.CLIP_VALUE)   # clip norm
            optimizer.step()                         # update parameters
            scheduler.step()                         # update lr

            t_b = time()

            if args.is_master:
                print(f'Step time consumption: {(t_b-t_a):.4f}s.')

                # if (step+1)%10 == 0:      # print current step training loss and accuracy for every 10 steps
                print(f'epoch {epoch}/{config.EPOCHS}, step {step+1}/{n_total_steps}, loss {train_loss.item():.4f}, acc {train_acc:.4f}, time {(t_b-t_a):.4f}s')


            train_epoch_loss += train_loss.item()    # sum training loss for current epoch
            train_epoch_acc += train_acc.item()      # sum training accuracy for current epoch


        # VALIDATION    
        with torch.no_grad():

            val_epoch_loss, val_epoch_acc = 0, 0
            model.eval()

            for step, batch in enumerate(val_dataloader):

                b_token_ids = batch['token_ids'].cuda(non_blocking=True)
                b_attention_masks = batch['attention_masks'].cuda(non_blocking=True)
                b_token_type_ids = batch['token_type_ids'].cuda(non_blocking=True)
                b_labels = batch['labels'].cuda(non_blocking=True)

                output = model(input_ids=b_token_ids, token_type_ids=b_token_type_ids, attention_mask=b_attention_masks, labels=b_labels)

                logits = output[1]

                val_loss = criterion(logits, b_labels)
                val_acc = multi_acc(logits, b_labels)
                
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        history['train_loss'].append(train_epoch_loss/len(train_dataloader))
        history['val_loss'].append(val_epoch_loss/len(val_dataloader))
        history['train_acc'].append(train_epoch_acc/len(train_dataloader))
        history['val_acc'].append(val_epoch_acc/len(val_dataloader))

        t2 = time()

        if args.is_master:                     
            print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_dataloader):.4f} | Val Loss: {val_epoch_loss/len(val_dataloader):.4f} | Train Acc: {train_epoch_acc/len(train_dataloader):.4f} | Val Acc: {val_epoch_acc/len(val_dataloader):.4f} | Time: {(t2-t1):.2f}s.')

            # Save model at the end of each epoch
            if get_epoch==0:
                torch.save(
                    {'model_state_dict': model.module.state_dict()},      # save ddp parameters
                    config.ROOT_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(epoch) + '.pt'    # set path
                )
            else:
                torch.save(
                    {'model_state_dict': model.module.state_dict()},      # save ddp parameters
                    config.ROOT_PATH + filenames[get_epoch].replace('.pt','') + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(epoch) + '.pt'   # set path
                )
 
    if args.is_master:                     
        print('Training finished!')

    # save history 
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.HISTORY_FILE, index=False)
    
    # save config
    copyfile('config.py', config.OUTPUT_CONFIG)

if __name__ == "__main__":
    # execute only if run as a script
    main()