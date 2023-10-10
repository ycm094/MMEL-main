"""
    top-k metric
"""
from circle_loss import cosine_similarity, dot_similarity
import torch
import numpy as np
from triplet_loss import TripletMarginLoss_Multi

def cal_top_k(args, query, pos_feats, search_feats):
    """
        Input query, positive sample features, negative sample features
        return the ranking of positive samples
        ------------------------------------------
        Args:
        Returns:
    """

    if args.similarity == 'cos':
        ans = similarity_rank(query, pos_feats, search_feats, cosine_similarity)
    elif args.similarity == 'dot':
        ans = similarity_rank(query, pos_feats, search_feats, dot_similarity)
    else:
        ans = lp_rank(query, pos_feats, search_feats, args.loss_p)

    return ans

def similarity_rank(query, pos_feats, search_feats, cal_sim):
    """
        Sample ranking based on similarity
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    
    sim_p = cal_sim(query, pos_feats).detach().cpu().numpy()  # batch_size, 1
    sim_s = cal_sim(query, search_feats).detach().cpu().numpy()  # batch_size, n_search
    
    sim = np.concatenate((sim_p, sim_s), -1)  #  sort [CN, N+1]
    sim_mat = sim - sim_p
    ranks = (sim_mat > 0).sum(-1) + 1


    return ranks, sim_p, sim_s



def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix


def cos_similar2(p, q):
    k, d = p.size()
    triplet_loss = TripletMarginLoss_Multi(d).to(p.device)

    p = p.unsqueeze(1).repeat(1, k, 1).view(-1, d)  # [k^2, d]  p[0] == p[1]
    q = q.unsqueeze(0).repeat(k, 1, 1).view(-1, d)  # [k^2, d]  q[0] != q[1]

    sim_matrix = triplet_loss.out_trans(torch.cat((p, q), -1))  # [k^2]
    sim_matrix = sim_matrix.view(k, k)
    return sim_matrix

    
def cal_top_k_mine_multi2(args, count, ent_num, prev_pos_feats, prev_search_feats, feats, with_multi_decoder=True):
    """
        Sample ranking based on similarity
        ------------------------------------------
        Args:
        count: [B]
        ent_num: [B]
        pos_feats: [B, CN, 2]
        search_feats: [B, CN, N, 2]
        Returns:
    """
    pos, neg = feats
    all_feats = torch.cat((pos.unsqueeze(-2), neg), -2)  # [B, CN, N+1, 2D]

    B, CN, N, L = prev_search_feats.size()
    pos_feats = prev_pos_feats # torch.softmax(prev_pos_feats, -1)
    search_feats = prev_search_feats # torch.softmax(prev_search_feats, -1)

    sim_p = pos_feats[:, :, 1].unsqueeze(-1) # .detach().cpu().numpy()  # batch_size, 1
    sim_s = search_feats[:, :, :, 1] # .detach().cpu().numpy()  # batch_size, n_search

    rank_list = []
    for i in range(len(count)):
        if count[i] == 0:
            continue
        num = ent_num[i]
        if num == 1:
            sim = torch.cat((sim_p[i][:num], sim_s[i][:num]), -1)  #  sort [CN, N+1]
            sim_mat = sim - sim_p[i][:num]
            ranks = (sim_mat > 0).sum(-1) + 1
            rank_list.append(ranks.item())
        else:
            # if num > 2:
            #     print(num)
            sim = torch.cat((sim_p[i][:num], sim_s[i][:num]), -1)  #  sort [CN, N+1]

            if with_multi_decoder:
                sim_s_tmp, sim_p_tmp = sim[0, 1:], sim[0, 0].unsqueeze(-1)
                sim_mat = sim[0] - sim_p_tmp
                ranks = (sim_mat > 0).sum(-1) + 1

            cn_max, _ = torch.max(sim, 1)
            sim_sort_idx = torch.argsort(cn_max, -1, descending=True)   # from large to small idx
            prev_count_ranks, cur_count_ranks = [], []
            count_feats, count_scores = [], []
            flag = False
            for j in range(len(sim_sort_idx)-1):   # for cn-dim
                if count_feats == []:
                    a = all_feats[i][sim_sort_idx[j]]
                    a_score = sim[sim_sort_idx[j]]
                else:
                    flag = True
                    a = torch.stack(count_feats, 0)
                    a_score = torch.tensor(count_scores).to(a.device)
                    count_feats, count_scores = [], []
                    prev_count_ranks = cur_count_ranks
                    cur_count_ranks = []
                b = all_feats[i][sim_sort_idx[j+1]]  # the sim_sort_idx[j+1]-th entity
                b_score = sim[sim_sort_idx[j+1]]
                sim_matrix = cos_similar2(a, b)
                sim_matrix += a_score.unsqueeze(1).repeat(1, N+1)
                sim_matrix += b_score.unsqueeze(0).repeat(N+1, 1)
                sim_matrix /= 3  # add
                score, idx_list = torch.sort(sim_matrix.view(-1), descending=True)
                for k in range(len(idx_list[:N+1])):
                    idx = idx_list[k]
                    x, y = int(idx / (N+1)), int(idx % (N+1))
                    count_feats.append(a[x] + b[y])
                    count_scores.append(sim_matrix[x][y].item())
                    if flag == False:    # the 1st one
                        cur_count_ranks.append((x, y))
                    else:
                        tmp = []
                        for t in prev_count_ranks[x]:
                            tmp.append(t)
                        tmp.append(y)
                        cur_count_ranks.append(tmp)

                assert len(count_feats) == N+1
            
            for j in range(len(sim_sort_idx)):   # for each CN-dim
                idx = sim_sort_idx[j].item()
                tmp = dict()
                for c_j in range(len(cur_count_ranks)):
                    c_idx = cur_count_ranks[c_j][idx]
                    if c_idx not in tmp:
                        tmp[c_idx] = count_scores[c_j]
                    else:
                        tmp[c_idx] = count_scores[c_j] if count_scores[c_j] > tmp[c_idx] else tmp[c_idx]
                for c_k in tmp.keys():
                    sim[idx][c_k] += tmp[c_k]

                sim_s_tmp, sim_p_tmp = sim[idx, 1:], sim[idx, 0].unsqueeze(-1)
                sim_mat = sim[idx] - sim_p_tmp
                ranks = (sim_mat > 0).sum(-1) + 1
                rank_list.append(ranks.item())
            
    return rank_list, sim_p.detach().cpu().numpy(), sim_s.detach().cpu().numpy()


def cal_top_k_mine(args, pos_feats, search_feats):
    """
        Input query, positive sample features, negative sample features
        return the ranking of positive samples
        ------------------------------------------
        Args:
        Returns:
    """

    if args.similarity == 'cos':
        ans = similarity_rank_mine(pos_feats, search_feats, cosine_similarity)
    elif args.similarity == 'dot':
        ans = similarity_rank_mine(query, pos_feats, search_feats, dot_similarity)
    else:
        ans = lp_rank(query, pos_feats, search_feats, args.loss_p)

    return ans


def similarity_rank_mine(pos_feats, search_feats, cal_sim):
    """
        Sample ranking based on similarity
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    B, N, L = search_feats.size()
    # pos_feats = torch.softmax(pos_feats, -1)
    # search_feats = torch.softmax(search_feats, -1)

    sim_p = pos_feats[:, 1].unsqueeze(-1).detach().cpu().numpy()  # batch_size, 1
    sim_s = search_feats[:, :, 1].detach().cpu().numpy()  # batch_size, n_search

    sim = np.concatenate((sim_p, sim_s), -1)  #  sort [CN, N+1]
    sim_mat = sim - sim_p
    # sim_mat = sim_s - sim_p
    ranks = (sim_mat > 0).sum(-1) + 1


    return ranks, sim_p, sim_s


def lp_distance(x, dim, p):
    return (x ** p).sum(dim=dim) ** (1 / p)


def lp_rank(query, pos_feats, search_feats, p=2):
    """
        Using LP distance to calculate the rank of positive examples
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    dis_p = lp_distance(query - pos_feats.squeeze(), dim=-1, p=p).detach().cpu().numpy()
    dis_sf = lp_distance(query.unsqueeze(1) - search_feats, dim=-1, p=p).detach().cpu().numpy()

    batch_size = dis_p.size(0)
    for i in range(batch_size):
        rank = 0
        for dis in dis_sf[i]:
            if dis < dis_p[i]:
                rank += 1
        rank_list.append(rank)

    return rank_list, dis_p, dis_sf