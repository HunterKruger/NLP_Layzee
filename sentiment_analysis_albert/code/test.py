import config
from model import Albert
from dataset import CustomDataset
from evaluation import MltClsEvaluation

import os
from time import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification

os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES    # specify GPU usage    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_labels(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    return y_pred_tags

def main():
    
    mode = input('Choose a mode (1 for evaluation, 2 for only inference):')
    mode = int(mode)
    
    ## Model init
    print('Initializing model and loading params...')
    file_dict = dict()
    for i, f in enumerate(os.listdir(config.ROOT_PATH)):
        if '.pt' in f:
            file_dict[i] = f
    print(file_dict)
    get_epoch = input('Choose an epoch:')
    get_epoch = int(get_epoch)
    print('Initialization and loading finished.')
    
    ## Loading and processing data
    print('Loading and processing test set...')
    if mode == 1:
        batch_size = config.BATCH_SIZE
        test_df = pd.read_csv(config.TEST_FILE)
        test_set = CustomDataset(
            sentences=test_df[config.CONTENT_FIELD].values.astype("str"),
            labels=test_df[config.LABEL_FIELD]
        )
    if mode == 2:
        f = open(config.ONLINE_FILE)
        lines = f.read()
        online_data = lines.split('\n')
        f.close()
        test_df = pd.DataFrame({config.CONTENT_FIELD: online_data})
        batch_size = config.ONLINE_BATCH_SIZE
        test_set = CustomDataset(sentences=test_df[config.CONTENT_FIELD].values.astype("str"))

    test_dataloader = DataLoader(
        dataset=test_set, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    count_sample = test_df.shape[0]
    count_batch = len(test_dataloader)
    print(f'Load {count_sample} samples in {count_batch} batches, batch size {batch_size}.') 
    print('Loading finished.')


    ## Init model
    print('Initializing model...')
    model = AlbertForSequenceClassification.from_pretrained(
        config.BASE_MODEL_PATH, 
        num_labels=config.CLASSES,
        output_attentions=False, 
        output_hidden_states=False
    )
    #model = Albert()
    print(model)
    checkpoint = torch.load(config.ROOT_PATH+file_dict[get_epoch], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if config.ONLY_CPU:
        print('Using CPU for inference.')
    else:                
        print('Using GPU for inference.')
        model = torch.nn.DataParallel(model)   
        model = model.cuda()

    # model.load_state_dict(checkpoint['model_state_dict'],False)  # for ddp
    print('Initialization finished.')

    ## Inference 
    t1 = time()

    local_labels = np.array([])

    with torch.no_grad():
        model.eval()
        for _, batch in enumerate(test_dataloader):

            if config.ONLY_CPU:
                b_token_ids = batch['token_ids']
                b_attention_masks = batch['attention_masks']
                b_token_type_ids = batch['token_type_ids']
            else:
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
            test_labels = get_labels(logits).cpu().numpy()
            local_labels = np.append(local_labels,test_labels)
    
    t2 = time()
    print(f'Inference finished in {(t2-t1):.2f}s, {(t2-t1)/count_batch:.4f}s for each batch, {(t2-t1)/count_sample:.4f}s for each sample.')


    # Evaluation
    if mode == 1:
        print('Starting evaluation...') 
        mle = MltClsEvaluation(local_labels, test_df.class_label, config.LABELS)
        mle.confusion_matrix()
        mle.detailed_metrics()  
        test_df['pred'] = local_labels   
        test_df.to_csv(config.OUTPUT_TEST, index=False)

    # save prediction
    if mode == 2:
        test_df['pred'] = local_labels
        test_df.to_csv(config.OUTPUT_ONLINE, index=False)
        print('File saved to '+config.OUTPUT_ONLINE+'.')
    

if __name__ == "__main__":
    # execute only if run as a script
    main()

