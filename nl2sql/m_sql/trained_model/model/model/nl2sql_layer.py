import torch
import torch.nn as nn
import torch.nn.functional as F

from sql_utils.utils_tableqa import *


class Seq2SQL_v1(nn.Module):
    def __init__(self, iS, dr):
        """
        iS: input size
        """
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.dr = dr

        self.cls_ff = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, iS, bias=True))
        self.column_tok_ff = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, iS, bias=True))

        self.W_select_num = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 1, bias=True), nn.Sigmoid())
        self.W_where_num_op = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 7, bias=True))

        self.W_select_col = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 1, bias=True), nn.Sigmoid())
        self.W_where_col = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 1, bias=True), nn.Sigmoid())

        self.W_select_agg = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 6, bias=True))
        self.W_where_op = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 4, bias=True))

        self.W_col_label = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 1, bias=True), nn.Sigmoid())
        self.match_nlu_ff = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, iS, bias=True))
        self.match_col_ff = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, iS, bias=True))
        self.W_cv_match = nn.Sequential(nn.Dropout(dr), nn.Linear(iS, 1, bias=True), nn.Sigmoid())

    def get_columns_representation(self, wemb_cls, wemb_hpu, l_hpu, l_hs):
        batch_size = wemb_cls.size(0)
        col_k = self.column_tok_ff(wemb_hpu)
        cls_seq = []
        for i, col_num in enumerate(l_hs):
            for k in range(col_num):
                cls_seq.append(wemb_cls[i].unsqueeze(0))
        wemb_cls_ex = torch.cat(cls_seq, dim=0).unsqueeze(1)
        cls_q_ex = self.cls_ff(wemb_cls_ex).transpose(1, 2)
        logits = torch.bmm(col_k, cls_q_ex).squeeze(2)
        mask = torch.zeros_like(logits).long().to(device)
        for i in range(mask.shape[0]):
            mask[i, :l_hpu[i]] = 1
        logits *= mask
        sum_logits = torch.sum(logits, dim=1).unsqueeze(1).expand(-1, logits.shape[1])
        attentions = logits / sum_logits
        col_emb = torch.sum(wemb_hpu * attentions.unsqueeze(2)\
                            .expand(-1, -1, wemb_hpu.shape[2]),
                            dim=1)
        col_emb += wemb_cls_ex.squeeze(1)

        # resize col_emb: (batch size, column num, embedding dim)
        max_col_num = max(l_hs)
        z = torch.zeros_like(col_emb[0]).unsqueeze(0)

        col_emb_final = torch.zeros((batch_size, max_col_num, col_emb.size(-1))).to(device).float()
        masks = torch.zeros((batch_size, max_col_num)).to(device).long()
        col_id = 0
        for b, col_num in enumerate(l_hs):
            col_emb_final[b, :col_num, :] = col_emb[col_id:col_id+col_num, :]
            masks[b, :col_num] = 1
            col_id += col_num

        assert col_id == sum(l_hs)
        return col_emb_final, masks

    def get_sub_columns(self, col_emb, index, nums, max_num=-1):
        max_num = max(nums) if max_num == -1 else max_num
        bS = col_emb.size(0)
        sub_columns = []
        mask = torch.zeros((bS, max_num)).long().to(device)
        for b, cols in enumerate(index):
            mask[b, :nums[b]] = 1
            real = [col_emb[b, col] for col in cols]
            pad = (max_num - nums[b]) * [col_emb[b, 0]]  # this padding could be wrong. Test with zero padding later.
            sub_columns1 = torch.stack(real + pad)  # It is not used in the loss function.
            sub_columns.append(sub_columns1)
        return torch.stack(sub_columns).to(device), mask

    def get_token_embeddings(self, wemb_n, l_n, t_to_tt_idx):

        t_to_tt_idx_tensor = self.pad_tensor(t_to_tt_idx, l_n) \
            .long().unsqueeze(-1).expand((-1, -1, wemb_n.shape[-1]))
        bS, seq_len, _ = t_to_tt_idx_tensor.shape
        token_emb = wemb_n.gather(1, t_to_tt_idx_tensor)
        mask = torch.zeros(bS, seq_len).to(device).long()
        for b, l in enumerate(l_n):
            mask[b, :l] = 1
        return token_emb, mask

    @staticmethod
    def pad_tensor(sequences, l_n=None):
        tensors = []
        if l_n is None:
            l_n = list(map(len, sequences))
        max_len = max(l_n)
        for b, seq in enumerate(sequences):
            tensors.append(
                torch.tensor(
                    seq + [0 for _ in range(max_len - l_n[b])]
                ).unsqueeze(0).to(device)
            )
        return torch.cat(tensors, dim=0)

    def get_val_embeddings(self, nlu_emb, val_indexes, val_nums, max_num=-1):
        max_val_num = max(val_nums) if max_num == -1 else max_num
        bS, seq_nax_len, dim = nlu_emb.shape
        val_emb = torch.zeros((bS, max_val_num, dim)).to(device)
        mask = torch.zeros((bS, max_val_num)).to(device).long()
        for b, (indexes, n) in enumerate(zip(val_indexes, val_nums)):
            mask[b, :n] = 1
            for i, (start, end) in enumerate(indexes):
                val_emb[b, i, :] = torch.sum(nlu_emb[b, start:end + 1, :], dim=0)\
                                       / (end - start + 1)
        return val_emb, mask

    def forward(self, wemb_cls, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, t_to_tt_idx,
                g_sn=None, g_sc=None, g_sa=None, g_wnop=None, g_wc=None,
                g_wo=None, g_wvi=None, g_tags=None, g_vm=None):

        cols_emb, col_mask = self.get_columns_representation(wemb_cls, wemb_hpu, l_hpu, l_hs)

        # select num
        s_sn = self.W_select_num(wemb_cls).squeeze(-1)
        pr_sn = g_sn if g_sn else pred_sn(s_sn)

        # select cols
        s_sc = self.W_select_col(cols_emb).squeeze(-1) * col_mask
        pr_sc = g_sc if g_sc else pred_sc(pr_sn, s_sc)

        # select aggs
        selected_cols_emb, selected_cols_mask = self.get_sub_columns(cols_emb, pr_sc, pr_sn)
        s_sa = self.W_select_agg(selected_cols_emb) \
               * selected_cols_mask.unsqueeze(-1).expand(-1, -1, 6)
        # pr_sa = g_sa if g_sa else pred_sa(pr_sn, s_sa)

        # cond num & conn op
        s_wnop = self.W_where_num_op(wemb_cls)
        pr_wnop = g_wnop if g_wnop else pred_wnop(s_wnop)
        pr_wn, pr_conn_op = [], []
        for nop in pr_wnop:
            conn_op, wn = wnop_map_reverse[nop]
            pr_wn.append(wn)
            pr_conn_op.append(conn_op)

        # where cols
        s_wc = self.W_where_col(cols_emb).squeeze(-1) * col_mask
        pr_wc = g_wc if g_wc else pred_wc(pr_wn, s_wc)

        # where col-op
        where_cols_emb, where_cols_mask = self.get_sub_columns(cols_emb, pr_wc, pr_wn, max_num=3)
        s_wo = self.W_where_op(where_cols_emb)
        # pr_wo = g_wo if g_wo else pred_wo(pr_wn, s_wo)

        # value tags
        token_emb, nlu_mask = self.get_token_embeddings(wemb_n, l_n, t_to_tt_idx)
        s_tags = self.W_col_label(token_emb).squeeze(-1) * nlu_mask
        pr_tags = self.pad_tensor(g_tags, l_n) if g_tags else (s_tags > 0.5).long()
        extracted_val, val_nums = extract_val(g_tags, l_n) if g_tags \
                                  else extract_val(pr_tags, l_n)

        # value match
        cols_m = self.match_col_ff(where_cols_emb)
        val_emb, val_mask = self.get_val_embeddings(token_emb, extracted_val, val_nums)
        vals_m = self.match_nlu_ff(val_emb)
        col_num = cols_m.size(1)
        val_num = vals_m.size(1)
        w_col_mask = where_cols_mask.unsqueeze(2).expand(-1, -1, val_num)
        w_val_mask = val_mask.unsqueeze(1).expand(-1, col_num, -1)
        s_match = self.W_cv_match(torch.tanh(cols_m.unsqueeze(2).expand(-1, -1, val_num, -1) +
                                             vals_m.unsqueeze(1).expand(-1, col_num, -1, -1)))\
                      .squeeze(-1) * w_col_mask * w_val_mask
        return s_sn, s_sc, s_sa, s_wnop, s_wc, s_wo, s_tags, s_match

