# this is for the custom metric with list states from https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torch
import math
from itertools import combinations
import numpy as np


class MySpearmanCorrCoef(Metric): # basic module class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(self.preds) # recommended to standardize the list of states to be a single concatenated tensor
        target = dim_zero_cat(self.target)
        # some intermediate computation...
        r_preds, r_target = _rank_data(preds), _rank_dat(target)
        preds_diff = r_preds - r_preds.mean(0)
        target_diff = r_target - r_target.mean(0)
        cov = (preds_diff * target_diff).mean(0)
        preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
        target_std = torch.sqrt((target_diff * target_diff).mean(0))
        # finalize the computations
        corrcoef = cov / (preds_std * target_std + eps)
        return torch.clamp(corrcoef, -1.0, 1.0)
    
class Recall(Metric): # basic module class
    def __init__(self, top_k, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.top_k = top_k

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self): 
        # this should be the same as the original recall_at_k function
        # we have list of states, in the original, they calculate this in the cpu.
        # parse inputs
        preds = dim_zero_cat(self.preds).detach().numpy() # recommended to standardize the list of states to be a single concatenated tensor
        target = dim_zero_cat(self.target).detach().numpy()
        sum_recall = 0.0
        num_users = len(preds) # the size(0)
        true_users = 0
        for i in range(num_users):
            act_set = set([target[i]])
            pred_set = set(preds[i][:self.top_k])
            if len(act_set) != 0:
                sum_recall += len(act_set & pred_set) / float(len(act_set))
                true_users += 1
        return torch.tensor(sum_recall / true_users)
    
class NDCG(Metric): # basic module class
    def __init__(self, top_k, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.top_k = top_k

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self): 
        # this should be the same as the original ndcg_k function
        # we have list of states, in the original, they calculate this in the cpu.
        # parse inputs
        preds = dim_zero_cat(self.preds).detach().numpy() # recommended to standardize the list of states to be a single concatenated tensor
        target = dim_zero_cat(self.target).detach().numpy()
        res = 0
        for user_id in range(len(target)):
            k = min(self.top_k, len([target[user_id]]))
            idcg = self.idcg_k(k)
            #print(int(preds[user_id][0]) in set([target[user_id]]),int(preds[user_id][0]))
            dcg_k = sum([int(preds[user_id][j] in
                            set([target[user_id]])) / math.log(j+2, 2) for j in range(self.top_k)])
            res += dcg_k / idcg
        return torch.tensor(res / float(len(target)))
    
    def idcg_k(self, k):
        res = sum([1.0/math.log(i+2, 2) for i in range(k)])
        if not res:
            return 1.0
        else:
            return res
        
class OIEMetric:
    '''
    Takes the item embeddings and then measures the AAE and SA
    '''
    def __init__(self, Z):
        self.Z = Z
        self.N = Z.shape[0]
        self.idx_pair_list = self.idx_pair_func() # get the pairwise of index
        self.max_dist = 0
        self.max_dist_func() # get the max distance

    def idx_pair_func(self):
        # get the combinations of two items
        idx_pair_list = list(combinations(range(self.N), 2))
        return idx_pair_list
    
    def dist_pair(self, x,y):
        return np.linalg.norm(x-y)
    
    def max_dist_func(self):
        for idx_pair in self.idx_pair_list:
            dist = self.dist_pair(self.Z[idx_pair[0]], self.Z[idx_pair[1]])
            if dist > self.max_dist:
                self.max_dist = dist

    def dist_pair_norm(self, x,y):
        return np.linalg.norm(x-y) / self.max_dist
    
    def AAE(self):
        # average absolute error
        aae = []
        for i in range(self.N - 1):
            aae.append(self.dist_pair_norm(self.Z[i], self.Z[i+1]))
        return np.mean(aae)
    
    def ordered_id_cond(self, zs, idxs):
        if len(idxs) != 3:
            raise ValueError("The length of the index should be 3")
        # fetch
        i, j, k = idxs[0], idxs[1], idxs[2]
        zi, zj, zk = zs[0], zs[1], zs[2]
        state = False # normally false
        # distance
        dij = self.dist_pair_norm(zi, zj)
        djk = self.dist_pair_norm(zj, zk)
        dik = self.dist_pair_norm(zi, zk)
        # print(dij, djk, dik)
        # constraint, just to make sure.q1
        if i>j or j>k or i>k:
            return state
        if i < j and j < k and i < k:
            if dij < dik and djk < dik:
                state = True
        return state
    
    def SA(self):
        # sequential adjacency
        sa = []
        for i in range(self.N - 2):
            sa.append(self.ordered_id_cond([self.Z[i], self.Z[i+1], self.Z[i+2]], [i, i+1, i+2]))
        return np.mean(sa)
     
def main():
    # Z = torch.randn(10, 5)
    a = np.random.randn(10)
    Z = np.array([a * i for i in range(1, 5001)]) # continuous vectors
    # print(Z)
    oie_metric = OIEMetric(Z)
    # print(len(oie_metric.idx_pair_list))
    # print(oie_metric.max_dist)
    from timeit import default_timer as timer
    start = timer()
    # print(oie_metric.AAE())
    print(oie_metric.SA())
    end = timer()
    print(end - start)

if __name__ == "__main__":
    main()