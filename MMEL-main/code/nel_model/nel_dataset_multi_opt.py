"""
    -----------------------------------
    dataset of nel
"""
import torch
from torch.utils.data import Dataset
import h5py
import json
import re
import random
from os.path import join

INAME_PATTERN = re.compile("/(\d+)\.")


def neg_sample_online(neg_id, neg_iid, tfidf_neg, negid2qid, max_sample_num=3, threshold=0.95):
    """
        Online negative sampling algorithm
        ------------------------------------------
        Args:
        Returns:
    """
    N = len(tfidf_neg)
    cands = set()

    while len(cands) < max_sample_num:
        rand = random.random()
        if not tfidf_neg[neg_id] or rand > threshold:
            cand = random.randint(0, N - 1)
        else:
            rand_word = random.choice(tfidf_neg[neg_id])
            cand = random.choice(neg_iid[rand_word])

        if cand != neg_id:
            cands.add(cand)

    return [negid2qid[c] for c in cands]


class NELDataset(Dataset):
    def __init__(self, args, guks,
                 all_input_ids,
                 all_input_mask,
                 all_segment_ids,
                 all_answer_id,
                 all_img_id,
                 all_mentions,
                 answer_list,
                 contain_search_res=False):
        # text info
        self.guks = guks
        self.guks2id =  {guk: i for i, guk in enumerate(self.guks)}
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_answer_id = all_answer_id
        self.all_img_id = all_img_id
        self.all_mentions = all_mentions

        # answer
        self.answer_list = answer_list  # id2ansStr
        self.answer_mapping = {answer: i for i, answer in enumerate(self.answer_list)}  # ansStr2id


        # Online negative sampling
        self.max_sample_num = args.neg_sample_num
        neg_config = json.load(open(args.path_neg_config))
        self.neg_iid = neg_config["neg_iid"]
        self.tfidf_neg = neg_config["tfidf_neg"]
        self.negid2qid = neg_config["keys_ordered"]
        self.qid2negid = {qid: i for i, qid in enumerate(neg_config["keys_ordered"])}

        # Sample features of negative sampling
        self.neg_list = json.load(open(join(args.dir_neg_feat, "neg_list.json")))
        self.neg_mapping = {sample: i for i, sample in enumerate(self.neg_list)}
        self.ansid2negid = {i: self.neg_mapping[ans] for i, ans in enumerate(self.answer_list)}

        neg_feat_h5 = h5py.File(join(args.dir_neg_feat, "neg_feats_2.h5"), 'r')
        self.neg_features = neg_feat_h5.get("features")
        
        # add
        neg_feat_h5 = h5py.File(join(args.dir_neg_feat, "neg_tokens.h5"), 'r')
        self.neg_tokens = neg_feat_h5.get("tokens")
        neg_feat_h5 = h5py.File(join(args.dir_neg_feat, "neg_mask.h5"), 'r')
        self.neg_masks = neg_feat_h5.get("mask")
        # add

        # image features
        img_list = json.load(open(join(args.dir_neg_feat, "img_list.json")))
        self.img_mapping = {iname: i for i, iname in enumerate(img_list)}

        img_feat_h5 = h5py.File(join(args.dir_neg_feat, "clip_img_feats.h5"), 'r')
        self.img_features = img_feat_h5.get("features")

        # entity image features
        ent_img_list = json.load(open(join(args.dir_neg_feat, "ent_img_list.json")))
        self.ent_img_mapping = {iname: i for i, iname in enumerate(ent_img_list)}

        ent_img_feat_h5 = h5py.File(join(args.dir_neg_feat, "clip_ent_img_feats.h5"), 'r')
        self.ent_img_features = ent_img_feat_h5.get("features")
        
        # search candidates
        self.contain_search_res = contain_search_res
        if self.contain_search_res:
            self.search_res = json.load(open(args.path_candidates, "r"))  # mention: [qid0, qid1, ..., qidn]

    def __len__(self):
        return len(self.all_answer_id)

    def set_train(self):
        self.train = True
        self.eval = False
        self.max_num = 2
    
    def set_eval(self):
        self.train = False
        self.eval = True
        self.max_num = 5
        
    def set_max_num(self, num):
        self.max_num = num
    
    def __getitem__(self, current_idx):
        sample = dict()
        idx_list = []
        guk = self.guks[current_idx]
        ent_num = -1
        flag = True
        if len(guk.split('-')) == 2:  # guk[-2] != '-':
            sample['single'] = 1
            ent_num = 1
            idx_list.append(current_idx)
            while len(idx_list) < self.max_num:
                if self.train:
                    idx_list.append(current_idx)
                else:
                    idx_list.append(-1)
        else:
            sample['single'] = 0
            k_id = int(guk[-1])   # 0, 1, 2, 3, 4
            if k_id > 0:
                flag = False
            begin_idx = current_idx - k_id
            guk = self.guks[begin_idx]
            while guk[-2] == '-':
                idx_list.append(begin_idx)
                begin_idx += 1
                if begin_idx >= len(self.guks):
                    break
                guk = self.guks[begin_idx]
                if guk[-1] == '0':
                    break

            if self.train:
                idx_list = random.sample(idx_list, self.max_num)
            else:
                ent_num = len(idx_list)
                while len(idx_list) < self.max_num:
                   idx_list.append(-1)
        
                # idx_list = []
                # idx_list.append(current_idx)

        cur_input_ids, cur_input_mask, cur_segment_ids, cur_answer_id = [], [], [], []
        pos_sample, pos_sample_tokens, pos_sample_mask = [], [], []
        neg_sample, neg_sample_tokens, neg_sample_mask = [], [], []
        all_pos_input_ids, all_pos_input_mask, all_neg_input_ids, all_neg_input_mask = [], [], [], []
        cur_search_res, cur_search_res_tokens, cur_search_res_mask = [], [], []
        cur_img_feats, cur_ent_img_feats, cur_neg_ent_img_feats, cur_search_neg_ent_img_feats = [], [], [], []
        for k in range(len(idx_list)):   
            idx = idx_list[k]
            if idx == -1:
                cur_input_ids.append(torch.zeros((cur_input_ids[-1].size())))
                cur_input_mask.append(torch.zeros((cur_input_mask[-1].size())))
                cur_segment_ids.append(torch.zeros((cur_segment_ids[-1].size())))
                cur_answer_id.append(torch.zeros((1)))

                cur_img_feats.append(torch.zeros((cur_img_feats[-1].size())))
                cur_ent_img_feats.append(torch.zeros((cur_ent_img_feats[-1].size())))
                cur_neg_ent_img_feats.append(torch.zeros((cur_neg_ent_img_feats[-1].size())))
                
                pos_sample.append(torch.zeros(pos_sample[-1].size()))
                pos_sample_tokens.append(torch.zeros(pos_sample_tokens[-1].size()))
                pos_sample_mask.append(torch.zeros(pos_sample_mask[-1].size()))

                neg_sample.append(torch.zeros(neg_sample[-1].size()))
                neg_sample_tokens.append(torch.zeros(neg_sample_tokens[-1].size()))
                neg_sample_mask.append(torch.zeros(neg_sample_mask[-1].size()))

                all_pos_input_ids.append(torch.zeros(all_pos_input_ids[-1].size()))
                all_pos_input_mask.append(torch.zeros(all_pos_input_mask[-1].size()))
                all_neg_input_ids.append(torch.zeros(all_neg_input_ids[-1].size()))
                all_neg_input_mask.append(torch.zeros(all_neg_input_mask[-1].size()))

                cur_search_res.append(torch.zeros(cur_search_res[-1].size()))
                cur_search_res_tokens.append(torch.zeros(cur_search_res_tokens[-1].size()))
                cur_search_res_mask.append(torch.zeros(cur_search_res_mask[-1].size()))
                cur_search_neg_ent_img_feats.append(torch.zeros(cur_search_neg_ent_img_feats[-1].size()))
                continue

            cur_input_ids.append(self.all_input_ids[idx])  # torch.tensor (hidden_size, )
            cur_input_mask.append(self.all_input_mask[idx])
            cur_segment_ids.append(self.all_segment_ids[idx])
            cur_answer_id.append(self.all_answer_id[idx])
            # sample["mentions"] = self.all_mentions[idx]

            # image
            img_id = self.all_img_id[idx]
            if img_id not in self.img_mapping.keys():
                img_id = self.img_mapping[img_id+'-0']
            else:
                img_id = self.img_mapping[img_id]
            cur_img_feats.append(torch.from_numpy(self.img_features[img_id]))

            ans_id = int(self.all_answer_id[idx])
            ans_str = self.answer_list[ans_id]
            # pos + neg samples
            pos_sample_id = self.ansid2negid[ans_id]
            pos_sample.append(torch.tensor([self.neg_features[pos_sample_id]]))
            pos_sample_tokens.append(torch.tensor([self.neg_tokens[pos_sample_id]]))
            pos_sample_mask.append(torch.tensor([self.neg_masks[pos_sample_id]]))

            # ent_image
            cur_ent_img_feats.append(torch.from_numpy(self.ent_img_features[pos_sample_id]))

            # Negative example：list
            neg_ids = neg_sample_online(self.qid2negid[ans_str], self.neg_iid, self.tfidf_neg, self.negid2qid, self.max_sample_num)
            neg_ids_map = [self.neg_mapping[nid] for nid in neg_ids]  # Convert negative example str id into id of sample feature
            neg_sample.append(torch.tensor([self.neg_features[nim] for nim in neg_ids_map]))
            neg_sample_tokens.append(torch.tensor([self.neg_tokens[nim] for nim in neg_ids_map]))
            neg_sample_mask.append(torch.tensor([self.neg_masks[nim] for nim in neg_ids_map]))

            # ent_image
            neg_sample_id = neg_ids_map[0]
            cur_neg_ent_img_feats.append(torch.from_numpy(self.ent_img_features[neg_sample_id]))

            # add
            # for positive sample
            input_tmp = self.all_input_ids[idx].clone()
            input_mask = self.all_input_mask[idx].clone()
            pos_tmp = torch.tensor([self.neg_tokens[pos_sample_id]])[0].int()
            pos_mask = torch.tensor([self.neg_masks[pos_sample_id]])[0].int()
            num = self.all_input_mask[idx].sum().item() - 1  # no -1 for not prompt
            if num > 90:
                input_tmp[91:128] = torch.tensor(1).to(input_tmp.device)
                input_mask[91:128] = torch.tensor(0).to(input_tmp.device)
                cur_input_mask[k][91:128] = torch.tensor(0).to(input_tmp.device)
                num = 91    
    
            input_tmp_pos = input_tmp.clone()
            input_mask_pos = input_mask.clone()

            num_pos = min((pos_tmp != 0).int().sum().item(), 129-num)
            input_tmp_pos[num: num+num_pos-1] = pos_tmp[1: num_pos]
            input_mask_pos[num: num+num_pos-1] = pos_mask[1: num_pos]
            # for i in range(num, 128):
            #     input_tmp[i] = pos_tmp[i-num+1]
            #     input_mask[i] = pos_mask[i-num+1]
            all_pos_input_ids.append(input_tmp_pos)  # torch.tensor (hidden_size, )
            all_pos_input_mask.append(input_mask_pos)
            # for negative sample
            # input_tmp = self.all_input_ids[idx].clone()
            # input_mask = self.all_input_mask[idx].clone()
            neg_tmp = torch.tensor([self.neg_tokens[nim] for nim in neg_ids_map])[0].int()
            neg_mask = torch.tensor([self.neg_masks[nim] for nim in neg_ids_map])[0].int()
            # num = self.all_input_mask[idx].sum().item() - 1
            num_neg = min((neg_tmp != 0).int().sum().item(), 129-num)
            
            input_tmp_neg = input_tmp.clone()
            input_mask_neg = input_mask.clone()
            input_tmp_neg[num: num+num_neg-1] = neg_tmp[1: num_neg]
            input_mask_neg[num: num+num_neg-1] = neg_mask[1: num_neg]
            
            all_neg_input_ids.append(input_tmp_neg)  # torch.tensor (hidden_size, )
            all_neg_input_mask.append(input_mask_neg)
            # add

            # return search results
            if self.contain_search_res:    
                qids_searched = self.search_res[self.all_mentions[idx]]
                qids_searched_map = [] # remove the correct answer
                flag_pos = False
                for qid in qids_searched:
                    if qid == ans_str and flag_pos == False:
                        flag_pos = True
                        continue
                    qids_searched_map.append(self.neg_mapping[qid])   
                if flag_pos == False:
                    qids_searched_map = qids_searched_map[:-1]
                # qids_searched_map = [self.neg_mapping[qid] for qid in qids_searched]
                cur_search_res.append(torch.tensor([self.neg_features[qsm] for qsm in qids_searched_map]))
                search_res_tokens = torch.tensor([self.neg_tokens[nim] for nim in qids_searched_map])
                search_res_mask = torch.tensor([self.neg_masks[nim] for nim in qids_searched_map])

                tmp, tmp_mask = [], []
                for j in range(9):
                # for j in range(len(qids_searched_map)):
                    # input_tmp = self.all_input_ids[idx].clone()
                    # input_mask = self.all_input_mask[idx].clone()
                    neg_tmp = search_res_tokens[j].int()
                    neg_mask = search_res_mask[j].int()
                    num_neg = min((neg_tmp != 0).int().sum().item(), 129-num)
                    input_tmp_neg = input_tmp.clone()
                    input_mask_neg = input_mask.clone()
                    input_tmp_neg[num: num+num_neg-1] = neg_tmp[1: num_neg]
                    input_mask_neg[num: num+num_neg-1] = neg_mask[1: num_neg]
            
                    tmp.append(input_tmp_neg)
                    tmp_mask.append(input_mask_neg)
                cur_search_res_tokens.append(torch.stack(tmp, 0))  # torch.tensor (hidden_size, )
                cur_search_res_mask.append(torch.stack(tmp_mask, 0))
                cur_search_neg_ent_img_feats.append(torch.stack([torch.from_numpy(self.ent_img_features[qsm]) for qsm in qids_searched_map], 0))
                

        sample["ent_num"] = ent_num
        sample["input_ids"] = torch.stack(cur_input_ids, 0)  # torch.tensor (hidden_size, )
        sample["input_mask"] = torch.stack(cur_input_mask, 0)
        sample["segment_ids"] = torch.stack(cur_segment_ids, 0)
        sample["answer_id"] = torch.tensor(cur_answer_id)
        sample["img_feat"] = torch.stack(cur_img_feats, 0)
        sample["ent_img_feat"] = torch.stack(cur_ent_img_feats, 0)
        sample["neg_ent_img_feat"] = torch.stack(cur_neg_ent_img_feats, 0)
        
        sample["pos_sample"] = torch.stack(pos_sample, 0)
        sample["pos_sample_tokens"] = torch.stack(pos_sample_tokens, 0)
        sample["pos_sample_mask"] = torch.stack(pos_sample_mask, 0)
        
        sample["neg_sample"] = torch.stack(neg_sample, 0)
        sample["neg_sample_tokens"] = torch.stack(neg_sample_tokens, 0)
        sample["neg_sample_mask"] = torch.stack(neg_sample_mask, 0)
        
        sample["all_pos_input_ids"] = torch.stack(all_pos_input_ids, 0)
        sample["all_pos_input_mask"] = torch.stack(all_pos_input_mask, 0)
        sample["all_neg_input_ids"] = torch.stack(all_neg_input_ids, 0)
        sample["all_neg_input_mask"] = torch.stack(all_neg_input_mask, 0)

        if self.contain_search_res:
            sample["search_res"] = torch.stack(cur_search_res, 0)
            sample["search_res_tokens"] = torch.stack(cur_search_res_tokens, 0)
            sample["search_res_mask"] = torch.stack(cur_search_res_mask, 0)
            sample["search_neg_ent_img_feats"] = torch.stack(cur_search_neg_ent_img_feats, 0)[:, :9]
        
        sample['count'] = 1 if flag else 0
        return sample
