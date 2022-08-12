import config
from model import Albert
from dataset import CustomDataset, process_data, train_test_split

import os
import datetime
from time import time 
from shutil import copyfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification
from transformers import get_polynomial_decay_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES      # specify GPU usage    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_model(get_epoch, epoch, model, filenames=None):
    if get_epoch==0:
        torch.save(
            {'model_state_dict': model.module.state_dict()},      # save parameters
            config.ROOT_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(epoch) + '.pt'    # set path
        )
    else:
        torch.save(
            {'model_state_dict': model.module.state_dict()},      # save parameters
            config.ROOT_PATH + filenames[get_epoch].replace('.pt','') + '__' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(epoch) + '.pt'   # set path
        )

def main():
    
    ## Loading data
    print('Loading dataset...')
    if config.ALREADY_SPLIT:
        train_df = pd.read_csv(config.TRAIN_FILE)
        val_df = pd.read_csv(config.VALIDATION_FILE)
        print('Training set shape: '+ str(train_df.shape))
        print('Validaiton set shape: '+ str(val_df.shape))
        print('Loading finished.')
    else:
        data_df = process_data(config.INPUT_FILE, config.CLS2IDX, True)     # DataFrame, only used labeled data
        train_df, test_df = train_test_split(
            data_df, 
            test_size=config.TEST_SIZE, 
            shuffle=True, 
            random_state=config.RANDOM_STATE)
        train_df, val_df = train_test_split(
            train_df, 
            test_size=config.VALIDATION_SIZE, 
            shuffle=True, 
            random_state=config.RANDOM_STATE)  
        print('Training set shape: '+ str(train_df.shape))
        print('Validaiton set shape: '+ str(val_df.shape))
        print('Test set shape: '+ str(test_df.shape))
        print('Loading finished.')
        print('Saving training set & validation set & test set to local...')
        train_df.to_csv(config.TRAIN_FILE, index=False)
        val_df.to_csv(config.VALIDATION_FILE, index=False)
        test_df.to_csv(config.TEST_FILE, index=False)
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
    filenames = dict()
    i = 1
    for f in os.listdir(config.ROOT_PATH):
        if '.pt' in f:
            filenames[i] = f
            i+=1
    if filenames:
        print(filenames)
        get_epoch = input('Choose a checkpoint (input 0 to train a new model):')
        get_epoch = int(get_epoch)
    else:
        get_epoch=0

    print('Initializing model...')

    ### model with downstream by huggingface
    model = AlbertForSequenceClassification.from_pretrained(
        config.BASE_MODEL_PATH, 
        num_labels=config.CLASSES,
        output_attentions=False, 
        output_hidden_states=False
    )           

    ### model with several layers' hidden states pooled and concatenated  
    # model = Albert()

    model = torch.nn.DataParallel(model.cuda())   # multi-gpu

    if get_epoch!=0:
        checkpoint = torch.load(config.ROOT_PATH+filenames[get_epoch])
        model.load_state_dict(checkpoint['model_state_dict'])
    print(model)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = params - trainable
    print(f'Parameters: {params}')
    print(f'Trainable: {trainable}')
    print(f'Untrainable: {untrainable}')
    print('Initialization finished.')


    ## Init tensorboard
    writer = SummaryWriter(log_dir=config.TENSORBOARD_FILE,flush_secs=config.TENSORBOARD_TIME)
    train_tensorboard_step  = 1
    val_tensorboard_step = 1

   
    ## Training and validation
    print('Start training...')
    if get_epoch == 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.LR, 
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=config.LR, 
            momentum=config.MOMENTUM, 
            nesterov=config.NESTEROV, 
            weight_decay=config.WEIGHT_DECAY
        )

    if config.SCHEDULER_STEPS is None:
        steps = len(train_dataloader) * config.EPOCHS
    else:
        steps = config.SCHEDULER_STEPS

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.WARMUP_STEPS, 
        num_training_steps=steps,
        lr_end=config.LR_END,
        power=config.POWER
    )   

    criterion = torch.nn.CrossEntropyLoss() 

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
            
            # print(output)
            # print(logits)
            # print(b_labels)
            
            train_loss = criterion(logits, b_labels)  # loss for current step
            train_acc = multi_acc(logits, b_labels)   # accuracy for current step
                
            optimizer.zero_grad()                    # clear gradients
            train_loss.backward()                    # back-propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.CLIP_NORM)   # clip gradients
            optimizer.step()                         # update parameters
            scheduler.step()                         # update lr

            t_b = time()

            if (step+1) % config.LOG_INTERVAL == 0:        # print current step training loss and accuracy for every LOG_INTERVAL steps
                print(f'epoch {epoch}/{config.EPOCHS}, step {step+1}/{n_total_steps}, loss {train_loss.item():.4f}, acc {train_acc:.4f}, time {(t_b-t_a):.4f}s')

            train_epoch_loss += train_loss.item()          # sum training loss for current epoch
            train_epoch_acc += train_acc.item()            # sum training accuracy for current epoch

            writer.add_scalar('Train_loss', train_loss, train_tensorboard_step)
            writer.add_scalar('Train_acc', train_acc, train_tensorboard_step)
            writer.add_scalar('LR', get_lr(optimizer), train_tensorboard_step)
            train_tensorboard_step += 1

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

                writer.add_scalar('Val_loss', val_loss, val_tensorboard_step)
                writer.add_scalar('Val_acc', val_acc, val_tensorboard_step)

                val_tensorboard_step += 1
        
        writer.add_scalar('Train_epoch_loss', train_epoch_loss/len(train_dataloader), epoch)
        writer.add_scalar('Val_epoch_loss'  , val_epoch_loss/len(val_dataloader)    , epoch)
        writer.add_scalar('Train_epoch_acc' , train_epoch_acc/len(train_dataloader) , epoch)
        writer.add_scalar('Val_epoch_acc'   , val_epoch_acc/len(val_dataloader)     , epoch)

        t2 = time()
        
        print(f'Epoch {epoch}: | Train Loss: {train_epoch_loss/len(train_dataloader):.4f} | Val Loss: {val_epoch_loss/len(val_dataloader):.4f} | Train Acc: {train_epoch_acc/len(train_dataloader):.4f} | Val Acc: {val_epoch_acc/len(val_dataloader):.4f} | Time: {(t2-t1):.2f}s.')

        # Early stopping
        current_loss = val_epoch_loss/len(val_dataloader)
        print(f'    Current epoch loss: {current_loss:.4f}')
        print(f'    Last epoch loss: {last_loss:.4f}')

        if epoch == config.EPOCHS:
            print('No early stopping, saving model...')
            save_model(get_epoch, epoch, model, filenames)

        if current_loss > last_loss:
            trigger_times += 1
            print('    Trigger times:', trigger_times)
            if trigger_times >= config.PATIENCE:
                print(f'Early stopping, saving the best model in epoch {best_epoch}!')
                save_model(get_epoch, best_epoch, best_model, filenames)
                break
        else:
            print('    Trigger times: 0')
            trigger_times = 0
            best_model = model
            best_epoch = epoch
        last_loss = current_loss

    print('Training finished!')

    # save config
    copyfile('config.py', config.OUTPUT_CONFIG)

    

if __name__ == "__main__":
    # execute only if run as a script
    main()

