import os
import argparse
import shutil
import sqlite3
import time

import tqdm
import torch
import random as python_random
from transformers import BertTokenizer, BertModel
import logging
import numpy as np
from model import Loss_sw_se, Seq2SQL_v1
import moxing as mox

from sql_utils.utils_tableqa import load_tableqa, get_loader, get_fields, get_g, get_g_wvi, get_wemb_bert, \
    pred_sw_se, convert_pr_wvi_to_string, generate_sql_i, extract_val, normalize_sql, get_acc, get_acc_x, \
    save_for_evaluation, load_tableqa_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hyper_param(parser):

    parser.add_argument("--eval", default='False', type=str)
    parser.add_argument("--no_save", default='False', type=str)
    parser.add_argument("--toy_model", default='False', type=str)
    parser.add_argument("--toy_size", default=16, type=int)

    parser.add_argument('--tepoch', default=15, type=int)
    parser.add_argument('--print_per_step', default=50, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default='False', type=str,
                        help="If present, BERT is trained.")

    parser.add_argument("--data_url", default='./data_and_model/tableqa_tok', type=str,
                        help="Saving path of model file, logfile and result file.")
    parser.add_argument("--train_url", default='./data_and_model/', type=str,
                        help="Saving path of model file, logfile and result file.")

    parser.add_argument("--vocab_file",
                        default='vocab.txt', type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--max_seq_length",
                        default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences ")
    parser.add_argument("--num_target_layers",
                        default=1, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--do_lower_case', default='False', type=str, help='whether to use lower case.')
    parser.add_argument("--bert_url", default='./m-sql/pre-trained_weights/chinese_wwm_ext_pytorch/', type=str,
                        help="Path or model name of BERT")
    parser.add_argument("--load_weight", default='./m-sql/trained_model/model/best_model.pth', type=str,
                        help="model path to load")
    parser.add_argument('--dr', default=0, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument('--num_warmup_steps', default=-1, type=int, help="num_warmup_steps")
    parser.add_argument("--split", default='val', type=str, help='prefix of jsonl and db files')

    args, _ = parser.parse_known_args()
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.do_lower_case = args.do_lower_case == 'True'
    args.fine_tune = args.fine_tune == 'True'
    args.no_save = args.no_save == 'True'
    args.eval = args.eval == 'True'
    args.toy_model = args.toy_model == 'True'

    return args


def get_bert(bert_path):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model_bert = BertModel.from_pretrained(bert_path)
    bert_config = model_bert.config
    model_bert.to(device)

    return model_bert, tokenizer, bert_config


def update_lr(param_groups, current_step, num_warmup_steps, start_lr):
    if current_step <= num_warmup_steps:
        warmup_frac_done = current_step / num_warmup_steps
        new_lr = start_lr * warmup_frac_done
        for param_group in param_groups:
            param_group['lr'] = new_lr


def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert


def get_models(args, logger, bert_model, trained=False, path_model=None, eval=False):
    # some constants
    if not eval:
        logger.info(f"Batch_size = {args.bS * args.accumulate_gradients}")
        logger.info(f"BERT parameters:")
        logger.info(f"learning rate: {args.lr_bert}")
        logger.info(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(bert_model)
    iS = bert_config.hidden_size * args.num_target_layers
    logger.info(bert_config.to_json_string())
    # Get Seq-to-SQL
    if not eval:
        logger.info(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
        logger.info(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(iS, args.dr)
    model = model.to(device)

    if trained:
        assert path_model != None
        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)
        model.load_state_dict(res['model'])
        model.to(device)

    return model, model_bert, tokenizer, bert_config


def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table = load_tableqa(path_wikisql, args.toy_model, args.toy_size,
                                                                no_hs_tok=True)
    train_loader, dev_loader = get_loader(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients, print_per_step, logger,
          current_step, st_pos=0, opt_bert=None):
    model.train()
    model_bert.train()
    torch.autograd.set_detect_anomaly(True)
    ave_loss = 0
    cnt = 0

    for iB, t in enumerate(train_loader):
        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.

        g_sn, g_sc, g_sa, g_wnop, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi, g_tags, g_value_match = get_g_wvi(t, g_wc)

        wemb_cls, wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        l_n_t = []
        for t in t_to_tt_idx:
            l_n_t.append(len(t))
        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.

        # score
        s_sn, s_sc, s_sa, s_wnop, s_wc, \
        s_wo, s_tags, s_match = model(wemb_cls, wemb_n, l_n_t, wemb_h, l_hpu, l_hs,
                                      t_to_tt_idx=t_to_tt_idx,
                                      g_sn=g_sn, g_sc=g_sc, g_sa=g_sa, g_wo=g_wo,
                                      g_wnop=g_wnop, g_wc=g_wc, g_wvi=g_wvi,
                                      g_tags=g_tags, g_vm=g_value_match)

        # Calculate loss & step
        loss = Loss_sw_se(s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match,
                          g_sn, g_sc, g_sa, g_wnop, g_wc, g_wo, g_tags, g_value_match)
        if iB % accumulate_gradients == 0:
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                update_lr(opt.param_groups, current_step, args.num_warmup_steps, args.lr)
                opt.step()
                if opt_bert:
                    update_lr(opt_bert.param_groups, current_step, args.num_warmup_steps, args.lr_bert)
                    opt_bert.step()
                current_step += 1
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            loss.backward()
            update_lr(opt.param_groups, current_step, args.num_warmup_steps, args.lr)
            opt.step()
            if opt_bert:
                update_lr(opt_bert.param_groups, current_step, args.num_warmup_steps, args.lr_bert)
                opt_bert.step()
            current_step += 1
        else:
            loss.backward()

        # statistics
        ave_loss += loss.item()

        if iB % print_per_step == 0:
            log = f'[Train Batch {iB}] '
            logs = []
            logs.append(f'average loss: {"%.4f" % (ave_loss / cnt,)}')
            logger.info(log + ', '.join(logs))

    ave_loss /= cnt
    return ave_loss, current_step


def test(data_loader, data_table, model, model_bert, bert_config, tokenizer, max_seq_length,
         num_target_layers, print_per_step, logger, path_db, st_pos=0):
    model.eval()
    model_bert.eval()

    cnt = 0
    cnt_sn = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wnop = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_lx = 0
    cnt_x = 0

    db_conn = sqlite3.connect(path_db)
    cursor = db_conn.cursor()
    results = []
    for iB, t in enumerate(data_loader):
        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        wemb_cls, wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        l_n_t = []
        for t in t_to_tt_idx:
            l_n_t.append(len(t))

        # score
        s_sn, s_sc, s_sa, s_wnop, s_wc, \
        s_wo, s_tags, s_match = model(wemb_cls, wemb_n, l_n_t, wemb_h, l_hpu, l_hs, t_to_tt_idx)

        # prediction
        pr_sn, pr_sc, pr_sa, pr_wn, pr_conn_op, \
        pr_wc, pr_wo, pr_tags, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match, l_n_t)
        pr_wv_str = convert_pr_wvi_to_string(pr_wvi, nlu_t)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_conn_op, pr_wc, pr_wo, pr_wv_str, nlu)
        value_indexes, value_nums = extract_val(pr_tags, l_n_t)

        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            normalize_sql(pr_sql_i1, tb[b])
            results1 = {}
            results1["sql"] = pr_sql_i1
            results1["gold_sql"] = sql_i[b]
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results1['value_indexes'] = value_indexes[b]
            results1['value_nums'] = value_nums[b]
            results1['pr_wc'] = pr_wc[b]
            sn, sc, sa, co, wn, wc, wo, wv, cond, sql = \
                get_acc(sql_i[b], pr_sql_i1, pr_wc[b], pr_wo[b], tb[b], normalized=True)
            cnt_sn += sn
            cnt_sc += sc
            cnt_sa += sa
            cnt_wnop += (co and wn)
            cnt_wc += wc
            cnt_wo += wo
            cnt_wv += wv

            cnt_lx += sql
            results1['correct'] = sql

            execution, res = get_acc_x(sql_i[b], pr_sql_i1, tb[b], cursor)
            cnt_x += execution
            results1['ex_correct'] = execution
            results1['result'] = res

            results.append(results1)

        # print acc
        cnts = [cnt_sn, cnt_sc, cnt_sa, cnt_wnop, cnt_wc,
                cnt_wo, cnt_wv, cnt_lx, cnt_x, (cnt_lx + cnt_x) / 2]
        cnt_desc = [
            's-num', 's-col', 's-col-agg', 'w-num-op', 'w-col',
            'w-col-op', 'w-col-value', 'acc_lx', 'acc_x', 'acc_mx'
        ]
        if iB % print_per_step == 0:
            log = f'[Test Batch {iB}] '
            logs = []
            for k, metric in enumerate(cnts):
                logs.append(cnt_desc[k] + ': ' + '%.4f' % (metric / cnt,))
            logger.info(log + ', '.join(logs))

    acc_sn = cnt_sn / cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wnop = cnt_wnop / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt
    acc_mx = (acc_lx + acc_x) / 2

    acc = [acc_sn, acc_sc, acc_sa, acc_wnop, acc_wc,
           acc_wo, acc_wv, acc_lx, acc_x, acc_mx]

    return acc, results, acc_lx


def print_result(epoch, acc, dname, logger=None):
    if logger:
        logger.info(f'------------ {dname} results ------------')
        if dname == 'dev':
            acc_sn, acc_sc, acc_sa, acc_wnop, acc_wc, \
            acc_wo, acc_wv, acc_lx, acc_x, acc_mx = acc
            logger.info(
                f" Epoch: {epoch}, s-num: {acc_sn:.4f}, s-col: {acc_sc:.4f},"
                f" s-col-agg: {acc_sa:.4f}, w-num-op: {acc_wnop:.4f},"
                f" w-col: {acc_wc:.4f}, w-col-op: {acc_wo:.4f}, w-col-value: {acc_wv:.4f},"
                f" acc_lx: {acc_lx:.4f}, acc_x: {acc_x:.4f}, acc_mx: {acc_mx:.4f}"
            )
        else:
            logger.info(f" Epoch: {epoch}, average loss: {acc}")


def get_logger(log_fp=None):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s')
    logger = logging.getLogger(__name__)
    if log_fp:
        handler = logging.FileHandler(log_fp)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def predict(data_loader, data_table, model, model_bert, bert_config, tokenizer,
            max_seq_length, num_target_layers, path_db):

    model.eval()
    model_bert.eval()

    results = []

    cnt = 0
    cnt_sn = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wnop = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_lx = 0
    cnt_x = 0

    db_conn = sqlite3.connect(path_db)
    cursor = db_conn.cursor()

    for iB, t in tqdm.tqdm(enumerate(data_loader)):
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)
        wemb_cls, wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        l_n_t = []
        for t in t_to_tt_idx:
            l_n_t.append(len(t))

        s_sn, s_sc, s_sa, s_wnop, s_wc, \
        s_wo, s_tags, s_match = model(wemb_cls, wemb_n, l_n_t, wemb_h, l_hpu, l_hs, t_to_tt_idx)
        # prediction
        pr_sn, pr_sc, pr_sa, pr_wn, pr_conn_op, \
        pr_wc, pr_wo, pr_tags, pr_wvi = pred_sw_se(s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match, l_n_t)
        pr_wv_str = convert_pr_wvi_to_string(pr_wvi, nlu_t)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_conn_op, pr_wc, pr_wo, pr_wv_str, nlu)
        value_indexes, value_nums = extract_val(pr_tags, l_n_t)

        for b, pr_sql_i1 in enumerate(pr_sql_i):
            cnt += 1
            results1 = {}
            normalize_sql(pr_sql_i1, tb[b])
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results1["sql"] = pr_sql_i1
            if sql_i[b]:
                results1["gold_sql"] = sql_i[b]
            results1['value_indexes'] = value_indexes[b]
            results1['value_nums'] = value_nums[b]
            results1['pr_wc'] = pr_wc[b]
            if sql_i[b]:
                sn, sc, sa, co, wn, wc, wo, wv, cond, sql =\
                    get_acc(sql_i[b], pr_sql_i1, pr_wc[b], pr_wo[b], tb[b], normalized=True)
                cnt_sn += sn
                cnt_sc += sc
                cnt_sa += sa
                cnt_wnop += (wn and co)
                cnt_wc += wc
                cnt_wo += wo
                cnt_wv += wv
                cnt_lx += sql
                results1['correct'] = sql

                execution, res = get_acc_x(sql_i[b], pr_sql_i1, tb[b], cursor)
                cnt_x += execution
                results1['ex_correct'] = execution
                results1['result'] = res

            results.append(results1)

    cnts = [cnt_sn, cnt_sc, cnt_sa, cnt_wnop, cnt_wc,
            cnt_wo, cnt_wv, cnt_lx, cnt_x, (cnt_x + cnt_lx) / 2]
    if sum(cnts) > 0:
        cnt_desc = [
            's-num', 's-col', 's-col-agg', 'w-num-op', 'w-col',
            'w-col-op', 'w-col-value', 'acc_lx', 'acc_x', 'acc_mx'
        ]

        logger.info('--------- eval result ---------')
        for k, metric in enumerate(cnts):
            logger.info(cnt_desc[k] + ': ' + '%.4f' % (metric / cnt,))
    else:
        cnts = None
        cnt_desc = None

    return results, cnt, cnts, cnt_desc


if __name__ == '__main__':

    # Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)
    save_path = args.train_url
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not args.eval:
        _model_path = './m-sql/trained_model/model/'
        shutil.copytree(_model_path, os.path.join(save_path, 'model'))

    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_fp = os.path.join(save_path, f'{t}.log')
    logger = get_logger(log_fp)
    logger.info(f"BERT-Model: {args.bert_url}")

    trained = args.load_weight is not None and args.load_weight != 'None'
    load_path = None
    if trained:
        load_path = '/home/work/modelarts/inputs/best_model.pt'
        if args.load_weight and args.load_weight.startswith('obs://'):
            if not os.path.exists(load_path):
                mox.file.copy_parallel(args.load_weight, load_path)
                print('copy %s to %s' % (args.load_weight, load_path))
            else:
                print(load_path, 'already exists')
        else:
            load_path = args.load_weight

    train_input_dir = args.data_url
    bert_model = args.bert_url

    # Paths
    path_wikisql = train_input_dir
    path_val_db = os.path.join(train_input_dir, 'val.db')
    path_save_for_evaluation = save_path

    # Build & Load models
    if args.eval and not trained:
        print('in eval mode, "--load_weight" must be provided!')
        exit(-1)

    if not trained:
        model, model_bert, tokenizer, bert_config = get_models(args, logger, bert_model, eval=args.eval)
    else:
        path_model = load_path
        model, model_bert, tokenizer, bert_config = get_models(args, logger, bert_model,
                                                               trained=True, path_model=path_model,
                                                               eval=args.eval)

    if not args.eval:

        train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
        opt, opt_bert = get_opt(model, model_bert, args.fine_tune)

        acc_lx_t_best = -1
        epoch_best = -1
        current_step = 1
        for epoch in range(args.tepoch):
            # train
            logger.info(f'Training Epoch {epoch}')
            ave_loss_train, current_step = train(train_loader,
                                                 train_table,
                                                 model,
                                                 model_bert,
                                                 opt,
                                                 bert_config,
                                                 tokenizer,
                                                 args.max_seq_length,
                                                 args.num_target_layers,
                                                 args.accumulate_gradients,
                                                 args.print_per_step,
                                                 logger=logger,
                                                 current_step=current_step,
                                                 opt_bert=opt_bert,
                                                 st_pos=0)

            # check DEV
            with torch.no_grad():
                logger.info(f'Testing on dev Epoch {epoch}:')
                acc_dev, results_dev, \
                    dev_acc_lx = test(dev_loader,
                                      dev_table,
                                      model,
                                      model_bert,
                                      bert_config,
                                      tokenizer,
                                      args.max_seq_length,
                                      args.num_target_layers,
                                      args.print_per_step,
                                      logger=logger,
                                      path_db=path_val_db,
                                      st_pos=0)

            print_result(epoch, ave_loss_train, 'train', logger=logger)
            print_result(epoch, acc_dev, 'dev', logger=logger)

            # save results for the official evaluation
            path_save_file = save_for_evaluation(path_save_for_evaluation,
                                                 results_dev, 'dev', epoch=epoch)
            # mox.file.copy_parallel(path_save_file,
            #                        args.train_url + f'results_dev-{epoch}.jsonl')

            # save best model
            # Based on Dev Set logical accuracy lx
            if dev_acc_lx > acc_lx_t_best:
                acc_lx_t_best = dev_acc_lx
                epoch_best = epoch
                # save model
                if not args.no_save:
                    state = {'model': model.state_dict(),
                             'model_bert': model_bert.state_dict()}
                    torch.save(state, os.path.join(save_path, 'model', f'best_model.pth'))

            logger.info(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
    else:
        try:
            dev_data, dev_table = load_tableqa_data(path_wikisql, mode=args.split, no_hs_tok=True)
        except Exception:
            logger.error('未找到输入文件！')
            exit(-1)
        dev_loader = torch.utils.data.DataLoader(
            batch_size=args.bS,
            dataset=dev_data,
            shuffle=False,
            num_workers=1,
            collate_fn=lambda x: x
        )
        with torch.no_grad():
            results, cnt, cnts, cnt_desc \
                    = predict(dev_loader,
                              dev_table,
                              model,
                              model_bert,
                              bert_config,
                              tokenizer,
                              args.max_seq_length,
                              args.num_target_layers,
                              os.path.join(train_input_dir, args.split + '.db'))
        save_for_evaluation(os.path.join(save_path, 'pred_results.jsonl'),
                            results, args.split, 'pred', use_filename=True)
        if cnts:
            with open(os.path.join(save_path, 'eval_result.txt'), 'w') as f_eval:
                f_eval.write('--------- eval result ---------\n')
                for k, metric in enumerate(cnts):
                    f_eval.write(cnt_desc[k] + ': ' + '%.4f' % (metric / cnt,) + '\n')


