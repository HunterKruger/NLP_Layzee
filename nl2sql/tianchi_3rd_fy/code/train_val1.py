"""
Created by FENG YUAN on 2021/10/6
"""

import config
from model1 import Model
from dataset import CustomDataset
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

import wandb
wandb.init(project="bert", entity="yuanfeng")

wandb.config = {
  "learning_rate": 0.00001,
  "epochs": 50,
  "batch_size": 32
}

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES  # specify GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
transformers.logging.set_verbosity_error()

def save_model(epoch, model):
    torch.save(
        {'model_state_dict': model.module.state_dict()},  # save parameters in single device format
        config.ROOT_MODEL_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + str(epoch) + '.pt' # set path
    )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mask_acc(logits, target, mask):
    """
    Accuracy with masks only for token classification.
    Args:
        logits: (batch, max_len, num_classes), not yet passed to Softmax 
        target: (batch, max_len) 
        mask  : (batch_size, max_len)
    Returns:
        Accuracy value masked by the length.
    """
    logits_softmax = torch.softmax(logits, dim=-1)         # (batch, max_len, num_classes)
    y_pred_tags    = torch.argmax(logits_softmax, dim=-1)  # (batch, max_len)
    correct_pred   = (y_pred_tags == target).float()       # (batch, max_len)
    correct_pred   = correct_pred * mask.float()           # (batch, max_len)
    acc            = correct_pred.sum() / torch.sum(mask)
    return acc


def masked_ce_loss(logits, target, mask):
    """
    Cross entropy loss only for token classification.
    Args:
        logits: (batch, max_len, num_classes), not yet passed to Softmax 
        target: (batch, max_len) 
        mask  : (batch_size, max_len)
    Returns:
        Averaged loss value masked by the length.
    """
    logits_softmax     = torch.softmax(logits, dim=-1)                # (batch, max_len, num_classes)
    logits_log_softmax = torch.log(logits_softmax)                    # (batch, max_len, num_classes)
    target             = torch.nn.functional.one_hot(target)          # (batch, max_len, num_classes)
    multi              = -torch.multiply(logits_log_softmax, target)  # (batch, max_len, num_classes)
    sum_multi          = multi.sum(-1)                                # (batch, max_len)
    sum_multi_masked   = sum_multi * mask                             # (batch, max_len)
    sum_multi_masked   = sum_multi_masked.sum()                       # (1,)
    sum_mask           = mask.sum()                                   # (1,)
    return sum_multi_masked / (sum_mask + 1e-9)


def main():
    ### Load data
    print('Loading data...')
    train_tables = read_tables(config.TRAIN_TABLE_FILE)
    train_data   = read_data(config.TRAIN_DATA_FILE, train_tables)
    val_tables   = read_tables(config.VAL_TABLE_FILE)
    val_data     = read_data(config.VAL_DATA_FILE, val_tables)
    print(f'{len(train_data)} samples and {len(train_tables)} tables in the training set')
    print(f'{len(val_data)}   samples and {len(val_tables)}   tables in the validation set')
    print('Loading finished.')

    ### Create dataset and dataloader
    print('Creating dataset and dataloader...')
    train_set = CustomDataset(train_data)
    val_set   = CustomDataset(val_data)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    n_total_steps = len(train_dataloader)
    print('Creating finished.')

    ### Init model
    print('Initializing model...')
    model = Model()
    print(model)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable = params - trainable
    print(f'Parameters: {params}, trainable: {trainable}, untrainable: {untrainable}')
    model = torch.nn.DataParallel(model.cuda())
    model = model.cuda()
    print('Initializing finished...')

    ## Init tensorboard
    writer = SummaryWriter(log_dir=config.TENSORBOARD_FILE, flush_secs=config.TENSORBOARD_TIME)
    train_tensorboard_step = 1

    ### Optimizer
    optimizer = torch.optim.RAdam(model.parameters(), lr=config.LR)

    ### Loss
    ce_loss = torch.nn.CrossEntropyLoss()

    ### Training & validation loop
    print('Start training...')

    # Early stopping parameters
    trigger_times = 0
    last_loss = 100
    best_model = None
    best_epoch = None

    for epoch in range(1, config.EPOCHS + 1):  # iterate epochs

        t_start_epoch = time()

        # TRAINING
        train_epoch_COND_CONN_OP_loss = 0
        train_epoch_SEL_AGG_loss      = 0
        train_epoch_COND_OP_loss      = 0
        train_epoch_loss              = 0

        train_epoch_COND_CONN_OP_acc = 0
        train_epoch_SEL_AGG_acc      = 0
        train_epoch_COND_OP_acc      = 0

        model.train()

        # iterate steps
        for step, batch in enumerate(train_dataloader):

            t_start_step = time()

            b_token_ids       = batch['token_ids'].cuda()
            b_attention_masks = batch['attention_masks'].cuda()
            b_header_ids      = batch['header_ids'].cuda()
            b_header_masks    = batch['header_masks'].cuda()

            b_COND_CONN_OP = batch['COND_CONN_OP'].cuda()  # shape (batch_size,)
            b_SEL_AGG      = batch['SEL_AGG'].cuda()       # shape (batch_size, max_headers)
            b_COND_OP      = batch['COND_OP'].cuda()       # shape (batch_size, max_headers)

            # forward-propagation
            output = model(
                token_ids=b_token_ids,
                attention_masks=b_attention_masks,
                header_ids=b_header_ids,
            )

            # logits for each subtask
            logits_COND_CONN_OP = output['COND_CONN_OP']  # shape (batch_size, COND_CONN_OP classes)
            logits_SEL_AGG      = output['SEL_AGG']       # shape (batch_size, max_headers, SEL_AGG classes)
            logits_COND_OP      = output['COND_OP']       # shape (batch_size, max_headers, COND_OP classes)

            # loss for current step
            train_COND_CONN_OP_loss = ce_loss(logits_COND_CONN_OP, b_COND_CONN_OP)
            train_SEL_AGG_loss      = masked_ce_loss(logits_SEL_AGG, b_SEL_AGG, b_header_masks)
            train_COND_OP_loss      = masked_ce_loss(logits_COND_OP, b_COND_OP, b_header_masks)
            train_loss              = train_SEL_AGG_loss + train_COND_OP_loss + train_COND_CONN_OP_loss

            # accuracy for current step
            train_COND_CONN_OP_acc = accuracy(logits_COND_CONN_OP, b_COND_CONN_OP)
            train_SEL_AGG_acc      = mask_acc(logits_SEL_AGG, b_SEL_AGG, b_header_masks)
            train_COND_OP_acc      = mask_acc(logits_COND_OP, b_COND_OP, b_header_masks)

            # clear gradients
            optimizer.zero_grad()

            # back-propagation
            train_loss.backward()
            # train_SEL_AGG_loss.backward(retain_graph=True)
            # train_COND_OP_loss.backward(retain_graph=True)
            # train_COND_CONN_OP_loss.backward()

            # update parameters
            optimizer.step()

            t_end_step = time()

            # print current step training loss and accuracy for every LOG_INTERVAL steps
            if (step + 1) % config.LOG_INTERVAL == 0:
                print(
                    f'epoch {epoch}/{config.EPOCHS}, ' +
                    f'step {step + 1}/{n_total_steps}, ' +
                    f'COND_CONN_OP loss {train_COND_CONN_OP_loss.item():.4f}, ' +
                    f'SEL_AGG loss {train_SEL_AGG_loss.item():.4f}, ' +
                    f'COND_CONN_OP loss {train_COND_OP_loss.item():.4f}, ' +
                    f'train loss {train_loss.item():.4f}, ' +
                    f'COND_CONN_OP acc {train_COND_CONN_OP_acc:.4f}, ' +
                    f'SEL_AGG acc {train_SEL_AGG_acc:.4f}, ' +
                    f'COND_OP acc {train_COND_OP_acc:.4f}, ' +
                    f'time {(t_end_step - t_start_step):.4f}s.'
                )

            # sum training loss for current epoch
            train_epoch_COND_CONN_OP_loss += train_COND_CONN_OP_loss.item()
            train_epoch_SEL_AGG_loss      += train_SEL_AGG_loss.item()
            train_epoch_COND_OP_loss      += train_COND_OP_loss.item()
            train_epoch_loss              += train_loss.item()

            # sum training accuracy for current epoch           
            train_epoch_COND_CONN_OP_acc += train_COND_CONN_OP_acc.item()
            train_epoch_SEL_AGG_acc      += train_SEL_AGG_acc.item()
            train_epoch_COND_OP_acc      += train_COND_OP_acc.item()

            # write training log to tensorboard by step
            writer.add_scalar('train_COND_CONN_OP_loss', train_COND_CONN_OP_loss, train_tensorboard_step)
            writer.add_scalar('train_SEL_AGG_loss'     , train_SEL_AGG_loss     , train_tensorboard_step)
            writer.add_scalar('train_COND_OP_loss'     , train_COND_OP_loss     , train_tensorboard_step)
            writer.add_scalar('train_loss'             , train_loss             , train_tensorboard_step)

            wandb.log({"train_COND_CONN_OP_loss": train_COND_CONN_OP_loss})
            wandb.log({"train_SEL_AGG_loss": train_SEL_AGG_loss})
            wandb.log({"train_COND_OP_loss": train_COND_OP_loss})
            wandb.log({"train_loss": train_loss})

            writer.add_scalar('train_COND_CONN_OP_acc', train_COND_CONN_OP_acc, train_tensorboard_step)
            writer.add_scalar('train_SEL_AGG_acc'     , train_SEL_AGG_acc     , train_tensorboard_step)
            writer.add_scalar('train_COND_OP_acc'     , train_COND_OP_acc     , train_tensorboard_step)

            writer.add_scalar('lr', get_lr(optimizer), train_tensorboard_step)

            train_tensorboard_step += 1

        # VALIDATION
        with torch.no_grad():

            val_epoch_COND_CONN_OP_loss = 0
            val_epoch_SEL_AGG_loss      = 0
            val_epoch_COND_OP_loss      = 0
            val_epoch_loss              = 0

            val_epoch_COND_CONN_OP_acc = 0
            val_epoch_SEL_AGG_acc      = 0
            val_epoch_COND_OP_acc      = 0

            model.eval()

            for step, batch in enumerate(val_dataloader):
                b_token_ids       = batch['token_ids'].cuda()
                b_attention_masks = batch['attention_masks'].cuda()
                b_header_ids      = batch['header_ids'].cuda()
                b_header_masks    = batch['header_masks'].cuda()

                b_COND_CONN_OP = batch['COND_CONN_OP'].cuda()  # shape (batch_size,)
                b_SEL_AGG      = batch['SEL_AGG'].cuda()  # shape (batch_size, max_headers)
                b_COND_OP      = batch['COND_OP'].cuda()  # shape (batch_size, max_headers)

                output = model(
                    token_ids=b_token_ids,
                    attention_masks=b_attention_masks,
                    header_ids=b_header_ids,
                )

                # logits for each subtask
                logits_COND_CONN_OP = output['COND_CONN_OP']  # shape (batch_size, COND_CONN_OP classes)
                logits_SEL_AGG      = output['SEL_AGG']       # shape (batch_size, max_headers, SEL_AGG classes)
                logits_COND_OP      = output['COND_OP']       # shape (batch_size, max_headers, COND_OP classes)

                val_COND_CONN_OP_loss = ce_loss(logits_COND_CONN_OP, b_COND_CONN_OP)
                val_SEL_AGG_loss      = masked_ce_loss(logits_SEL_AGG, b_SEL_AGG, b_header_masks)
                val_COND_OP_loss      = masked_ce_loss(logits_COND_OP, b_COND_OP, b_header_masks)
                val_loss              = val_COND_CONN_OP_loss + val_SEL_AGG_loss + val_COND_OP_loss

                val_COND_CONN_OP_acc = accuracy(logits_COND_CONN_OP, b_COND_CONN_OP)
                val_SEL_AGG_acc      = mask_acc(logits_SEL_AGG, b_SEL_AGG, b_header_masks)
                val_COND_OP_acc      = mask_acc(logits_COND_OP, b_COND_OP, b_header_masks)

                val_epoch_COND_CONN_OP_loss += val_COND_CONN_OP_loss.item()
                val_epoch_SEL_AGG_loss      += val_SEL_AGG_loss.item()
                val_epoch_COND_OP_loss      += val_COND_OP_loss.item()
                val_epoch_loss              += val_loss.item()
            
                val_epoch_COND_CONN_OP_acc += val_COND_CONN_OP_acc.item()
                val_epoch_SEL_AGG_acc      += val_SEL_AGG_acc.item()
                val_epoch_COND_OP_acc      += val_COND_OP_acc.item()

        t_end_epoch = time()

        print('**********************************************************************************************')
        print(f'Epoch {epoch + 0:03}:')
        print()
        print(f'    Train COND_CONN_OP Loss  : {train_epoch_COND_CONN_OP_loss / len(train_dataloader):.4f}')
        print(f'    Val COND_CONN_OP Loss    : {val_epoch_COND_CONN_OP_loss   / len(val_dataloader):.4f}')
        print(f'    Train COND_CONN_OP Acc   : {train_epoch_COND_CONN_OP_acc  / len(train_dataloader):.4f}')
        print(f'    Val COND_CONN_OP Acc     : {val_epoch_COND_CONN_OP_acc    / len(val_dataloader):.4f}')
        print()
        print(f'    Train SEL_AGG Loss       : {train_epoch_SEL_AGG_loss / len(train_dataloader):.4f}')
        print(f'    Val SEL_AGG Loss         : {val_epoch_SEL_AGG_loss   / len(val_dataloader):.4f}')
        print(f'    Train SEL_AGG Acc        : {train_epoch_SEL_AGG_acc  / len(train_dataloader):.4f}')
        print(f'    Val SEL_AGG Acc          : {val_epoch_SEL_AGG_acc    / len(val_dataloader):.4f}')
        print()
        print(f'    Train COND_OP loss       : {train_epoch_COND_OP_loss / len(train_dataloader):.4f}')
        print(f'    Val COND_OP loss         : {val_epoch_COND_OP_loss   / len(val_dataloader):.4f}')
        print(f'    Train COND_OP Acc        : {train_epoch_COND_OP_acc  / len(train_dataloader):.4f}')
        print(f'    Val COND_OP Acc          : {val_epoch_COND_OP_acc    / len(val_dataloader):.4f}')
        print()
        print(f'    Train Loss  : {train_epoch_loss / len(train_dataloader):.4f}')
        print(f'    Val Loss    : {val_epoch_loss   / len(val_dataloader):.4f}')
        print()
        print(f'Time: {(t_end_epoch - t_start_epoch):.2f}s.')
        print('**********************************************************************************************')

        wandb.log({"train_epoch_COND_CONN_OP_loss": train_epoch_COND_CONN_OP_loss / len(train_dataloader)})
        wandb.log({"val_epoch_COND_CONN_OP_loss": val_epoch_COND_CONN_OP_loss / len(train_dataloader)})
        wandb.log({"train_epoch_COND_CONN_OP_acc": train_epoch_COND_CONN_OP_acc / len(train_dataloader)})
        wandb.log({"val_epoch_COND_CONN_OP_acc": val_epoch_COND_CONN_OP_acc / len(train_dataloader)})

        wandb.log({"train_epoch_SEL_AGG_loss": train_epoch_SEL_AGG_loss / len(train_dataloader)})
        wandb.log({"val_epoch_SEL_AGG_loss": val_epoch_SEL_AGG_loss / len(train_dataloader)})
        wandb.log({"train_epoch_SEL_AGG_acc": train_epoch_SEL_AGG_acc / len(train_dataloader)})
        wandb.log({"val_epoch_SEL_AGG_acc": val_epoch_SEL_AGG_acc / len(train_dataloader)})

        wandb.log({"train_epoch_COND_OP_loss": train_epoch_COND_OP_loss / len(train_dataloader)})
        wandb.log({"val_epoch_COND_OP_loss": val_epoch_COND_OP_loss / len(train_dataloader)})
        wandb.log({"train_epoch_COND_OP_acc": train_epoch_COND_OP_acc / len(train_dataloader)})
        wandb.log({"val_epoch_COND_OP_acc": val_epoch_COND_OP_acc / len(train_dataloader)})

        wandb.log({"Train_Loss": train_epoch_loss / len(train_dataloader)})
        wandb.log({"Val_Loss": val_epoch_loss / len(train_dataloader)})


        # write training and validation log to tensorboard by epoch
        writer.add_scalar('train_epoch_COND_CONN_OP_loss', train_epoch_COND_CONN_OP_loss / len(train_dataloader), epoch)
        writer.add_scalar('train_epoch_COND_CONN_OP_acc' , train_epoch_COND_CONN_OP_acc  / len(train_dataloader), epoch)
        writer.add_scalar('val_epoch_COND_CONN_OP_loss'  , val_epoch_COND_CONN_OP_loss   / len(val_dataloader)  , epoch)
        writer.add_scalar('val_epoch_COND_CONN_OP_acc'   , val_epoch_COND_CONN_OP_acc    / len(val_dataloader)  , epoch)

        writer.add_scalar('train_epoch_SEL_AGG_loss', train_epoch_SEL_AGG_loss / len(train_dataloader), epoch)
        writer.add_scalar('train_epoch_SEL_AGG_acc' , train_epoch_SEL_AGG_acc  / len(train_dataloader), epoch)
        writer.add_scalar('val_epoch_SEL_AGG_loss'  , val_epoch_SEL_AGG_loss   / len(val_dataloader)  , epoch)
        writer.add_scalar('val_epoch_SEL_AGG_acc'   , val_epoch_SEL_AGG_acc    / len(val_dataloader)  , epoch)

        writer.add_scalar('train_epoch_COND_OP_loss', train_epoch_COND_OP_loss / len(train_dataloader), epoch)
        writer.add_scalar('train_epoch_COND_OP_acc' , train_epoch_COND_OP_acc  / len(train_dataloader), epoch)
        writer.add_scalar('val_epoch_COND_OP_loss'  , val_epoch_COND_OP_loss   / len(val_dataloader)  , epoch)
        writer.add_scalar('val_epoch_COND_OP_acc'   , val_epoch_COND_OP_acc    / len(val_dataloader)  , epoch)

        writer.add_scalar('train_epoch_loss', train_epoch_loss / len(train_dataloader), epoch)
        writer.add_scalar('val_epoch_loss'  , val_epoch_loss   / len(val_dataloader)  , epoch)

        # Early stopping
        current_loss = val_epoch_loss/len(val_dataloader)
        print(f'    Current epoch loss: {current_loss:.4f}')
        print(f'    Last epoch loss: {last_loss:.4f}')

        if epoch == config.EPOCHS:
            print('    No early stopping, saving model...')
            save_model(epoch, model)
            break

        if current_loss > last_loss:
            trigger_times += 1
            print('    Trigger times:', trigger_times)
            if trigger_times >= config.PATIENCE:
                print(f'Early stopping, saving the best model in epoch {best_epoch}!')
                save_model(best_epoch, best_model)
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
