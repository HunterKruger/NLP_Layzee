import config
from model import MSQL
from dataset import CustomDataset, collate_fn
from utils import read_data, read_tables

import os 
import datetime
from shutil import copyfile
from time import time 

import torch
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader
import transformers

from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES      # specify GPU usage  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
transformers.logging.set_verbosity_error()


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():

    ### Load data
    print('Loading data...')
    train_tables = read_tables(config.train_table_file)
    train_data = read_data(config.train_data_file, train_tables)
    val_tables = read_tables(config.val_table_file)
    val_data = read_data(config.val_data_file, val_tables)
    print(f'{len(train_data)} samples and {len(train_tables)} tables in the training set')
    print(f'{len(val_data)} samples and {len(val_tables)} tables in the validation set')
    print('Loading finished.')

    # Filter data
    print('Filtering data...')
    new_train_data = []
    new_val_data = []
    for sample in train_data:
        W_num_op_label = sample.sql.conn_sql_dict[sample.sql.cond_conn_op] + '-' + str(len(sample.sql.conds)) 
        S_num_label = len(sample.sql.sel)
        if S_num_label!=3 and W_num_op_label!='OR-4' and W_num_op_label!='AND-4' and W_num_op_label!='OR-3' and W_num_op_label!='AND-3':
            new_train_data.append(sample)
    for sample in val_data:
        W_num_op_label = sample.sql.conn_sql_dict[sample.sql.cond_conn_op] + '-' + str(len(sample.sql.conds)) 
        S_num_label = len(sample.sql.sel)
        if S_num_label!=3 and W_num_op_label!='OR-4' and W_num_op_label!='AND-4' and W_num_op_label!='OR-3' and W_num_op_label!='AND-3':
            new_val_data.append(sample)
    print(f'{len(new_train_data)} samples and {len(train_tables)} tables in the training set')
    print(f'{len(new_val_data)} samples and {len(val_tables)} tables in the training set')
    train_data = new_train_data
    val_data = new_val_data
    print('Filtering finished.')

    ### Create dataset and dataloader
    print('Creating dataset and dataloader...')
    train_set = CustomDataset(train_data)
    val_set = CustomDataset(val_data)
    train_dataloader = DataLoader(
        dataset=train_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    n_total_steps = len(train_dataloader)  
    print('Creating finished.')

    ### Init model
    print('Initializing model...')      
    model = MSQL()
    print(model)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = params - trainable
    print(f'Parameters: {params}, trainable: {trainable}, untrainable: {untrainable}')
    model = torch.nn.DataParallel(model.cuda())   
    model = model.cuda()
    print('Initializing finished...')   

    ## Init tensorboard
    writer = SummaryWriter(log_dir=config.TENSORBOARD_FILE,flush_secs=config.TENSORBOARD_TIME)
    train_tensorboard_step  = 1

    ### Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LR
    )   

    ### Loss
    criterion = torch.nn.CrossEntropyLoss() 

    ### Training & validation loop
    print('Start training...')
    for epoch in range(1, config.EPOCHS+1):   # iterate epochs
        
        t_start_epoch = time()

        # TRAINING
        train_epoch_S_sum_loss = 0
        train_epoch_W_num_op_loss = 0
        train_epoch_S_sum_acc = 0
        train_epoch_W_num_op_acc = 0

        model.train()

        # iterate steps
        for step, batch in enumerate(train_dataloader):  
            
            t_start_step = time()

            b_token_ids = batch['token_ids'].cuda()
            b_attention_masks = batch['attention_masks'].cuda()
            b_token_type_ids = batch['token_type_ids'].cuda()
            b_SEP_ids = batch['SEP_ids'].cuda()
            b_SEP_masks = batch['SEP_masks'].cuda()
            b_question_masks = batch['question_masks'].cuda()

            b_S_num = batch['S_num'].cuda()                      # shape (batch_size,)
            b_W_num_op = batch['W_num_op'].cuda()                # shape (batch_size,)

            # forward-propagation
            output = model( 
                token_ids=b_token_ids, 
                token_type_ids=b_token_type_ids, 
                attention_masks=b_attention_masks,
                SEP_ids=b_SEP_ids,
                SEP_masks=b_SEP_masks,
                question_masks=b_question_masks
            )

            # logits for each subtask
            logits_S_sum = output['S_num']       # shape (batch_size, S_num classes)
            logits_W_num_op = output['W_num_op']    # shape (batch_size, W_num_op classes)

            # loss for current step
            train_S_sum_loss = criterion(logits_S_sum, b_S_num)     
            train_W_num_op_loss = criterion(logits_W_num_op, b_W_num_op)     

            # accuracy for current step
            train_S_sum_acc = accuracy(logits_S_sum, b_S_num)   
            train_W_num_op_acc = accuracy(logits_W_num_op, b_W_num_op)   
                
            # clear gradients
            optimizer.zero_grad()                   

            # back-propagation
            train_W_num_op_loss.backward(retain_graph=True)              
            train_S_sum_loss.backward()              

            # update parameters
            optimizer.step()      

            t_end_step = time()

            # print current step training loss and accuracy for every LOG_INTERVAL steps
            if (step+1) % config.LOG_INTERVAL == 0:        
                print(
                    f'epoch {epoch}/{config.EPOCHS}, ' +
                    f'step {step+1}/{n_total_steps}, ' +
                    f'S_num loss {train_S_sum_loss.item():.4f}, ' +
                    f'W_num_op loss {train_W_num_op_loss.item():.4f}, ' +
                    f'S_sum acc {train_S_sum_acc:.4f}, ' +
                    f'W_num_op acc {train_W_num_op_acc:.4f}, ' +
                    f'time {(t_end_step - t_start_step):.4f}s.'
                )

            # sum training loss for current epoch
            train_epoch_S_sum_loss += train_S_sum_loss.item()                
            train_epoch_W_num_op_loss += train_W_num_op_loss.item()  

            # sum training accuracy for current epoch           
            train_epoch_S_sum_acc += train_S_sum_acc.item()                  
            train_epoch_W_num_op_acc += train_W_num_op_acc.item()     

            # write training log to tensorboard by step
            writer.add_scalar('train_S_sum_loss', train_S_sum_loss, train_tensorboard_step)
            writer.add_scalar('train_W_num_op_loss', train_W_num_op_loss, train_tensorboard_step)
            writer.add_scalar('train_S_sum_acc', train_S_sum_acc, train_tensorboard_step)
            writer.add_scalar('train_W_num_op_acc', train_S_sum_acc, train_tensorboard_step)
            writer.add_scalar('lr', get_lr(optimizer), train_tensorboard_step)
            train_tensorboard_step += 1      


        # VALIDATION    
        with torch.no_grad():

            val_epoch_S_sum_loss = 0
            val_epoch_W_num_op_loss = 0
            val_epoch_S_sum_acc = 0
            val_epoch_W_num_op_acc = 0

            model.eval()

            for step, batch in enumerate(val_dataloader):

                b_token_ids = batch['token_ids'].cuda()
                b_attention_masks = batch['attention_masks'].cuda()
                b_token_type_ids = batch['token_type_ids'].cuda()
                b_SEP_ids = batch['SEP_ids'].cuda()
                b_SEP_masks = batch['SEP_masks'].cuda()
                b_question_masks = batch['question_masks'].cuda()

                b_W_num_op = batch['W_num_op'].cuda()
                b_S_num = batch['S_num'].cuda()                     

                output = model(
                    token_ids=b_token_ids, 
                    token_type_ids=b_token_type_ids, 
                    attention_masks=b_attention_masks,
                    SEP_ids=b_SEP_ids,     
                    SEP_masks=b_SEP_masks,
                    question_masks=b_question_masks
                )

                # logits for each subtask
                logits_S_sum = output['S_num']
                logits_W_num_op = output['W_num_op']

                val_S_sum_loss = criterion(logits_S_sum, b_S_num)
                val_W_num_op_loss = criterion(logits_W_num_op, b_W_num_op)

                val_S_sum_acc = accuracy(logits_S_sum, b_S_num)
                val_W_num_op_acc = accuracy(logits_W_num_op, b_W_num_op)

                val_epoch_S_sum_loss += val_S_sum_loss.item()
                val_epoch_W_num_op_loss += val_W_num_op_loss.item()

                val_epoch_S_sum_acc += val_S_sum_acc.item()
                val_epoch_W_num_op_acc += val_W_num_op_acc.item()


        t_end_epoch = time()
        
        print('**********************************************************************************************')
        print(f'Epoch {epoch+0:03}:')
        print(f'    Train S_sum Loss: {train_epoch_S_sum_loss/len(train_dataloader):.4f}')
        print(f'    Val S_sum Loss: {val_epoch_S_sum_loss/len(val_dataloader):.4f}')          
        print(f'    Train W_num_op Loss: {train_epoch_W_num_op_loss/len(train_dataloader):.4f}')
        print(f'    Val W_num_op Loss: {val_epoch_W_num_op_loss/len(val_dataloader):.4f}')
        print(f'    Train S_sum Acc: {train_epoch_S_sum_acc/len(train_dataloader):.4f}')
        print(f'    Val S_sum Acc: {val_epoch_S_sum_acc/len(val_dataloader):.4f}')
        print(f'    Train W_num_op Acc: {train_epoch_W_num_op_acc/len(train_dataloader):.4f}')
        print(f'    Val W_num_op Acc: {val_epoch_W_num_op_acc/len(val_dataloader):.4f}')
        print(f'Time: {(t_end_epoch-t_start_epoch):.2f}s.')
        print('**********************************************************************************************')

        # write training and validation log to tensorboard by epoch
        writer.add_scalar('train_epoch_S_sum_loss', train_epoch_S_sum_loss/len(train_dataloader), epoch)
        writer.add_scalar('train_epoch_W_num_op_loss', train_epoch_W_num_op_loss/len(train_dataloader), epoch)
        writer.add_scalar('val_epoch_S_sum_loss', val_epoch_S_sum_loss/len(val_dataloader), epoch)       
        writer.add_scalar('val_epoch_W_num_op_loss', val_epoch_W_num_op_loss/len(val_dataloader), epoch)
        writer.add_scalar('train_epoch_S_sum_acc', train_epoch_S_sum_acc/len(train_dataloader), epoch)
        writer.add_scalar('train_epoch_W_num_op_acc', train_epoch_W_num_op_acc/len(train_dataloader), epoch)
        writer.add_scalar('val_epoch_S_sum_acc', val_epoch_S_sum_acc/len(val_dataloader), epoch)
        writer.add_scalar('val_epoch_W_num_op_acc', val_epoch_W_num_op_acc/len(val_dataloader), epoch)

        # Save model at the end of each epoch
        torch.save(
            {'model_state_dict': model.module.state_dict()},                                                         # save parameters in single device format
            config.ROOT_MODEL_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(epoch) + '.pt'    # set path
        )
   
    print('Training finished!')

    # save config
    copyfile('config.py', config.OUTPUT_CONFIG)

if __name__ == "__main__":
    # execute only if run as a script
    main()
