import torch
import torch.nn as nn
# from torch.nn import TripletMarginLoss
from word_level import WordLevel, ImgLevel
from phrase_level import PhraseLevel
from sent_level import SentLevel
from gated_fuse import GatedFusion
from recursive_encoder import RecursiveEncoder
from circle_loss import CircleLoss
from triplet_loss import TripletMarginLoss, TripletMarginLoss_Multi

def Contrastive_loss(out_1, out_2, batch_size, temperature=0.5):
    out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)  # [2*B, 2*B]
    '''
    torch.mm是矩阵乘法，a*b是对应位置上的数相除，维度和a，b一样
    '''
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    '''
    torch.eye生成对角线上为1，其他为0的矩阵
    torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    '''
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1)-pos_sim))).mean()
    return loss

class NELModel(nn.Module):
    def __init__(self, args):
        super(NELModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.output_size = args.output_size
        self.seq_len = args.max_sent_length
        self.text_feat_size = args.text_feat_size
        self.img_feat_size = self.hidden_size  # args.img_feat_size
        self.feat_cate = args.feat_cate.lower()

        self.resnet = False
        k = 4
        if self.resnet:
            self.img_trans = nn.Sequential(
                nn.Linear(self.hidden_size*4, self.hidden_size),  #
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
                nn.Dropout(self.dropout)
            )
        else:
            self.img_trans = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size*k),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size*k),
                nn.Dropout(self.dropout)
            )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )
        if 'w' in self.feat_cate:
            self.word_level = WordLevel(args)
        if 'p' in self.feat_cate:
            self.phrase_level = PhraseLevel(args)
        if 's' in self.feat_cate:
            self.sent_level = SentLevel(args)

        self.img_level = ImgLevel(args)

        self.gated_fuse = GatedFusion(args)

        # Dimension reduction
        self.out_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size*2, 2),
        )

        # Dimension reduction
        self.img_out_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size*2, 2),
        )

        # circle loss
        self.loss_function = args.loss_function
        self.loss_margin = args.loss_margin
        self.sim = args.similarity
        if self.loss_function == 'circle':
            self.loss_scale = args.loss_scale
            self.loss = CircleLoss(self.loss_scale, self.loss_margin, self.sim)
        elif self.loss_function == 'triplet':
            self.loss_p = args.loss_p
            self.loss = TripletMarginLoss(margin=self.loss_margin, p=self.loss_p)
        else:
            self.loss = nn.CrossEntropyLoss()
            self.loss_p = args.loss_p
            self.triplet_loss = TripletMarginLoss_Multi(hidden_size=self.hidden_size, margin=self.loss_margin, p=self.loss_p)

    def fusion(self, seq_feat, img_trans, mask): # [B, L, D], [B, L, D], [B, L]
        # seq_feat = fusion
        seq_att_w, img_att_w, attn_w, seq_att_p, img_att_p, seq_att_s, img_att_s, attn_p = 0, 0, 0, 0, 0, 0, 0, 0

        if 'w' in self.feat_cate:
            seq_att_w, img_att_w, attn_w = self.word_level(seq_feat, img_trans, mask)

        if 'p' in self.feat_cate:
            seq_att_p, img_att_p, context_feat, attn_p = self.phrase_level(seq_feat, img_trans, mask)

        if 's' in self.feat_cate:
            seq_att_s, img_att_s, attn_s = self.sent_level(context_feat, img_trans)
        fusion = self.gated_fuse(seq_att_w, img_att_w, seq_att_p, img_att_p, seq_att_s,img_att_s)
        return fusion

    def mention(self, B, CN, L, bert_mask, fusion, img_trans):
        mask = bert_mask.unsqueeze(-2).repeat(1, fusion.size(1), 1).view(-1, 1, L)
        fusion = fusion.view(-1, L, self.hidden_size)  

        fusion = self.fusion(fusion, img_trans, mask)
        fusion = fusion.view(B*CN, -1, self.hidden_size)
        pos_fusion, neg_fusion = fusion[:, 0], fusion[:, 1:].squeeze()
        return pos_fusion, neg_fusion
    
    def entity(self, B, CN, L, bert_mask, fusion, img_trans):
        mask = bert_mask.view(-1, 1, L)
        fusion = fusion.view(-1, L, self.hidden_size)  

        fusion = self.fusion(fusion, img_trans, mask)
        fusion = fusion.view(B*CN, -1, self.hidden_size)
        pos_fusion, neg_fusion = fusion[:, 0], fusion[:, 1:].squeeze()
        return pos_fusion, neg_fusion

    def text_entity(self, B, CN, L, bert_mask, fusion):
        mask = bert_mask.view(-1, L)
        fusion = fusion.view(-1, L, self.hidden_size)  
        
        fusion = (fusion * mask.unsqueeze(-1).repeat(1, 1, fusion.size(2))).sum(1) / mask.unsqueeze(-1).repeat(1, 1, fusion.size(2)).sum(1)
        fusion = self.trans(fusion)

        fusion = fusion.view(B*CN, -1, self.hidden_size)
        pos_fusion, neg_fusion = fusion[:, 0], fusion[:, 1:].squeeze()
        return pos_fusion, neg_fusion

    def forward(self, bert_feat, img, bert_mask=None, pos_feats=None, neg_feats=None, batch=None, mode='train', multi=False, ent_img_flag=True):
        """

            ------------------------------------------
            Args:
                bert_feat: tensor: (batch_size, max_seq_len, text_feat_size), the output of bert hidden size
                img: float tensor: (batch_size, ..., img_feat_size), image features - resnet
                bert_mask: tensor: (batch_size, max_seq_len)
                pos_feats(optional): (batch_size, n_pos, output_size)
                neg_feats(optional): (batch_size, n_neg, output_size)
            Returns:
        """
        if len(bert_feat[1].size()) == 5:
            B, CN, N, L, D = bert_feat[1].size()  # CN = num of contrastive samples
        else:
            B, CN, L, D = bert_feat[1].size()  # CN = num of contrastive samples
            N = 1

        if self.resnet:
            img = img.view(B, 2048, -1).permute(0, 2, 1)
            
        img_trans = self.img_trans(img).squeeze()
        img_trans = img_trans.view((B*CN, -1, self.img_feat_size))

        if ent_img_flag:
            ent_img = batch['ent_img_feat']  
            if self.resnet:
                ent_img = ent_img.view(B, 2048, -1).permute(0, 2, 1)

            ent_img_trans_pos = self.img_trans(ent_img).squeeze()
            ent_img_trans_pos = ent_img_trans_pos.view((B*CN, -1, self.img_feat_size))
        
        # review all the tensors
        bert_feat0 = bert_feat[0].view(B*CN, L, D)
        bert_feat1 = bert_feat[1].view(-1, L, D)
        bert_mask = bert_mask.view(B*CN, L)
        batch['all_pos_input_mask'] = batch['all_pos_input_mask'].view(B*CN, L)
        # review all the tensors
        pos_bert_trans = self.text_trans(bert_feat0)
        neg_bert_trans = self.text_trans(bert_feat1)
        
        # add
        if mode == 'train':
            assert 0 not in bert_mask.sum(-1)

        pos_fusion = pos_bert_trans 
        pos_img_trans, ent_img_trans_pos = self.img_level(img_trans, ent_img_trans_pos)
        
        if len(bert_feat[1].size()) == 4:  # [B, L, D]
            neg_fusion = neg_bert_trans
            fusion = torch.cat((pos_fusion.unsqueeze(1), neg_fusion.unsqueeze(1)), 1)
            batch['all_neg_input_mask'] = batch['all_neg_input_mask'].view(B*CN, L)
            neg_mask = (batch['all_neg_input_mask'] != bert_mask).int()
            neg_feats = (neg_bert_trans * neg_mask.unsqueeze(-1).repeat(1, 1, neg_bert_trans.size(2))).sum(1) / neg_mask.unsqueeze(-1).repeat(1, 1, neg_bert_trans.size(2)).sum(1)
            neg_feats_trans = self.trans(neg_feats)
            ent_img = batch['neg_ent_img_feat'] 
            if self.resnet:
                ent_img = ent_img.view(B, 2048, -1).permute(0, 2, 1)

            ent_img_trans_neg = self.img_trans(ent_img).squeeze()
            ent_img_trans_neg = ent_img_trans_neg.view(B*CN, -1, self.img_feat_size)
        
            neg_img_trans, ent_img_trans_neg = self.img_level(img_trans, ent_img_trans_neg)
            img_trans = torch.cat((pos_img_trans.unsqueeze(1), neg_img_trans.unsqueeze(1)), 1).view(-1, pos_img_trans.size(1), self.hidden_size)
        elif len(bert_feat[1].size()) == 5:  # [B, N, L, D]
            neg_bert_trans = neg_bert_trans.view(B*CN, N, L, -1)
            mask = bert_mask.unsqueeze(1).unsqueeze(-1).repeat(1, neg_bert_trans.size(1), 1, neg_bert_trans.size(-1))
            neg_fusion = neg_bert_trans 
            fusion = torch.cat((pos_fusion.unsqueeze(1), neg_fusion), 1)
            
            neg_bert_mask = bert_mask.unsqueeze(1).repeat(1, N, 1)
            batch['search_res_mask'] = batch['search_res_mask'].view(B*CN, N, L)
            neg_mask = (batch['search_res_mask'] != neg_bert_mask).int()
            mask = neg_mask.unsqueeze(-1).repeat(1, 1, 1, neg_bert_trans.size(-1))
            neg_feats_trans = (neg_bert_trans * mask).sum(-2) / mask.sum(-2)
            
            if self.resnet:
                ent_img = batch['search_neg_ent_img_feats'].view((B*CN, N, 2048, 7*7))
                ent_img = ent_img.permute(0, 1, 3, 2)
            else:
                ent_img = batch['search_neg_ent_img_feats'].view((B*CN, N, -1))
            
            ent_img_trans_neg = self.img_trans(ent_img).squeeze()
            img_trans = img_trans.unsqueeze(1).repeat(1, N, 1, 1)
            neg_img_trans, ent_img_trans_neg = self.img_level(img_trans.view(B*CN*N, -1, self.img_feat_size), ent_img_trans_neg.view(B*CN*N, -1, self.img_feat_size))
            img_trans = torch.cat((pos_img_trans.unsqueeze(1), neg_img_trans.view(B*CN, N, -1, self.img_feat_size)), 1).view(-1, pos_img_trans.size(1), self.hidden_size)
        
        pos_fusion, neg_fusion = self.mention(B, CN, L, bert_mask, fusion, img_trans)
        
        pos_mask = (batch['all_pos_input_mask'] != bert_mask).int()
        if len(bert_feat[1].size()) == 4:  # [B, L, D]
            entity_mask = torch.cat((pos_mask.unsqueeze(1), neg_mask.unsqueeze(1)), 1) # [B, 1+1, L]
            img_trans = torch.cat((ent_img_trans_pos.unsqueeze(1), ent_img_trans_neg.unsqueeze(1)), 1).view(-1, pos_img_trans.size(1), self.hidden_size)
        elif len(bert_feat[1].size()) == 5:  # [B, L, D]    
            entity_mask = torch.cat((pos_mask.unsqueeze(1), neg_mask), 1)  # [B, N+1, L]
            img_trans = torch.cat((ent_img_trans_pos.unsqueeze(1), ent_img_trans_neg.view(B*CN, N, -1, self.img_feat_size)), 1).view(-1, pos_img_trans.size(1), self.hidden_size)
        
        pos_feats_trans, neg_feats_trans = self.entity(B, CN, L, entity_mask, fusion, img_trans)

        if self.loss_function == 'circle':
            loss = self.loss(query, pos_feats_trans, neg_feats_trans)
        elif self.loss_function == 'triplet':
            loss = self.loss(query, pos_feats_trans.squeeze(), neg_feats_trans.squeeze())
        else:
            pos_score = self.out_trans(torch.cat((pos_fusion, pos_feats_trans), -1))
            neg_score = self.out_trans(torch.cat((neg_fusion, neg_feats_trans), -1))
            pos_label, neg_label = torch.ones(B*CN).to(pos_fusion.device), torch.zeros(B*CN*N).to(pos_fusion.device)
            loss_pos = self.loss(pos_score, pos_label.long())
            loss_neg = self.loss(neg_score.view(B*CN*N, -1), neg_label.long())
            loss = loss_pos + loss_neg
            if mode != 'train':
                pos_score = torch.softmax(pos_score, -1)
                neg_score = torch.softmax(neg_score, -1)

            # for contrastive loss
            if mode == 'train' and multi == True:
                pos_feats_trans = pos_feats_trans.view(B, CN, -1)
                neg_feats_trans = neg_feats_trans.view(B, CN, -1)
                pos_feats_1 = pos_feats_trans[:, 1] 
                pos_feats_2 = pos_feats_trans[:, 0] 
                neg_feats_1 = neg_feats_trans[:, 1] 
                neg_feats_2 = neg_feats_trans[:, 0] 
                # triplet_loss = self.triplet_loss(pos_feats_1, pos_feats_2, neg_feats_1, neg_feats_2) 
                triplet_loss = self.triplet_loss(pos_feats_2, pos_feats_1, neg_feats_1) 
                triplet_loss += self.triplet_loss(pos_feats_1, pos_feats_2, neg_feats_2)
                triplet_loss *= 0.5
                loss += triplet_loss
            else:
                pos_feats = pos_feats_trans.view(B, CN, -1) 
                neg_feats = neg_feats_trans.view(B, CN, N, -1)  
                triplet_loss = 0
            # for contrastive loss

        return (loss, triplet_loss), (pos_score, neg_score), (pos_feats, neg_feats)

    def trans(self, x):
        return x



