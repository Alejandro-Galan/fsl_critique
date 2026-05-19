from typing import Tuple

import torch, sys
import torch.nn.functional as F
import numpy as np

from torch.nn.modules import Module
import importlib
from utils import constants
from utils.constants import Const_c
# Initialize reading the json constants file for each experiment
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)
Constants = Constants_c.Constants
from torch.autograd import Variable

def invariance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(za, zb)


def variance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    std_za = torch.sqrt(za.var(dim=0) + eps)
    std_zb = torch.sqrt(zb.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_za)) + torch.mean(F.relu(1 - std_zb))
    return std_loss


def off_diagonal(z: torch.Tensor) -> torch.Tensor:
    # Return a flattened view of the off-diagonal elements of a square matrix
    n, m = z.shape
    assert n == m
    diag = torch.eye(n, device=z.device)
    return z[~diag.bool()]


def covariance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    N, D = za.shape
    za = za - za.mean(dim=0)
    zb = zb - zb.mean(dim=0)
    cov_za = (za.T @ za) / (N - 1)
    cov_zb = (zb.T @ zb) / (N - 1)
    cov_loss = (
        off_diagonal(cov_za).pow_(2).sum() / D + off_diagonal(cov_zb).pow_(2).sum() / D
    )
    return cov_loss


def vicreg_loss(
    za: torch.Tensor,
    zb: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sim_loss = sim_loss_weight * invariance_loss(za, zb)
    var_loss = var_loss_weight * variance_loss(za, zb)
    cov_loss = cov_loss_weight * covariance_loss(za, zb)
    loss = sim_loss + var_loss + cov_loss
    return {
        "loss": loss,
        "sim_loss": sim_loss,
        "var_loss": var_loss,
        "cov_loss": cov_loss,
    }



### From PrototypicalNetworks

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


# def euclidean_dist(x, y):
#     '''
#     Compute euclidean distance between two tensors
#     '''
#     # x: N x D
#     # y: M x D



#     n = x.size(1)
#     m = y.size(1)
#     dim = x.size(2)
#     b = x.size(0)
#     if dim != y.size(2) or b != y.size(0):
#         raise Exception

#     x = x.unsqueeze(2).expand(b, n, m, dim)
#     y = y.unsqueeze(1).expand(b, n, m, dim)


#     return torch.pow(x - y, 2).sum(3)


    



# def select_query_idx(q_idx, input):
#     all_q = []
#     for idx_list in q_idx:  
#         for i in idx_list:
#             if len(tuple(i)) < 1:
#                 breakpoint()
#         q_class = torch.stack([input[tuple(sample[0])] for sample in idx_list])
#         all_q.append(q_class)

#     return torch.stack(all_q)

# def select_proto_idx(support_idxs, input_cpu, batch_size):
#     all_protos = []
#     for idx_list in support_idxs:   
#         proto_example = []
#         for b in range(batch_size):
#             proto_batch = []
#             for sample in idx_list:
#                 tuple_s = tuple(sample)
#                 if tuple_s[0] == b:
#                     proto_batch.append(input_cpu[tuple_s])
        
#             proto_class = torch.stack(proto_batch)
#             proto_example.append(proto_class.mean(0))

#         proto_example = torch.stack(proto_example)
#         all_protos.append(proto_example)

#     prototypes = torch.stack(all_protos)
#     return prototypes


# def ORIGINALprototypical_loss(self, sample):
#     xs = Variable(sample['xs']) # support
#     xq = Variable(sample['xq']) # query

#     n_class = xs.size(0)
#     assert xq.size(0) == n_class
#     n_support = xs.size(1)
#     n_query = xq.size(1)

#     target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
#     target_inds = Variable(target_inds, requires_grad=False)

#     if xq.is_cuda:
#         target_inds = target_inds.cuda()

#     x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
#                     xq.view(n_class * n_query, *xq.size()[2:])], 0)

#     z = self.encoder.forward(x)
#     z_dim = z.size(-1)

#     z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
#     zq = z[n_class*n_support:]

#     dists = euclidean_dist(zq, z_proto)

#     log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

#     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

#     _, y_hat = log_p_y.max(2)
#     acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

#     return loss_val, {
#         'loss': loss_val.item(),
#         'acc': acc_val.item()
#     }



def euclidean_dist(x, y):
    """Compute squared Euclidean distances between rows of two tensors."""
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be two-dimensional tensors.")
    if x.size(1) != y.size(1):
        raise ValueError("x and y must have the same feature dimension.")

    return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=2)

def proto_loss_batch(input_cpu, target_cpu, n_support):

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support, 0]

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    try:
        query_idxs = torch.stack(
            [target_cpu.eq(c).nonzero()[n_support:, 0] for c in classes]
        ).view(-1)
    except RuntimeError as exc:
        raise ValueError("Could not build query indexes for prototypical loss.") from exc

    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    return dists

def prototypical_loss(input, target, n_support, samples_per_class, batch_size):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu').squeeze(1)
    input_cpu = input.to('cpu')

    all_dists = []
    for b in range(batch_size):
        all_dists.append(proto_loss_batch(input_cpu[b], target_cpu[b], n_support))
    dists = torch.stack(all_dists)
    n_classes = dists.shape[1]
    log_p_y = F.log_softmax(-dists, dim=2).view(batch_size, n_classes, Constants["n_query"], -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(1, n_classes, 1, 1)
    target_inds = target_inds.expand(batch_size, n_classes, Constants["n_query"], 1).long()

    loss_val = -log_p_y.gather(3, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(3)
    acc_val = y_hat.eq(target_inds.squeeze(3)).float().mean()

    return acc_val, loss_val, dists




# def prototypical_loss(input, target, n_support, samples_per_class, batch_size):
#     '''
#     Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

#     Compute the barycentres by averaging the features of n_support
#     samples for each class in target, computes then the distances from each
#     samples' features to each one of the barycentres, computes the
#     log_probability for each n_query samples for each one of the current
#     classes, of appartaining to a class c, loss and accuracy are then computed
#     and returned
#     Args:
#     - input: the model output for a batch of samples
#     - target: ground truth for the above batch of samples
#     - n_support: number of samples to keep in account when computing
#       barycentres, for each one of the current classes
#     '''
#     target_cpu = target.to('cpu')
#     input_cpu = input.to('cpu')     

#     def supp_idxs(c):
#         # FIXME when torch will support where as np
#         aux_ids = []
#         for b in range(batch_size):
#             subl = target_cpu[b].eq(c).nonzero()[:samples_per_class].squeeze(1).tolist()
#             # Add the batch idx
#             aux_ids += [[b] + pos[:-1] for pos in subl]
#         # return target_cpu.eq(c).nonzero()[:samples_per_class].squeeze(1)

#         return torch.tensor(aux_ids)
#         #### CHANGE: only one query in one particular class
#         # return target_cpu.eq(c).nonzero()[:samples_per_class].squeeze(1)

#     def query_idxs(c):
#         aux_ids = []
#         for b in range(batch_size):
#             subl = target_cpu[b].eq(c).nonzero()[samples_per_class:].squeeze(1).tolist()
#             # FineTuning scenario
#             if len(subl) < Constants["n_query:
#                 subl = target_cpu[b].eq(c).nonzero()[:n_query].squeeze(1).tolist()
#             # Add the batch idx
#             aux_ids.append([[b] + pos[:-1] for pos in subl])

#         return torch.tensor(aux_ids)
    

#     # FIXME when torch.unique will be available on cuda too
#     classes = torch.unique(target_cpu)
#     n_classes = len(classes)
#     # FIXME when torch will support where as np
#     # assuming n_query, n_target constants
#     #### CHANGE: only one query in one particular class
#     n_query = Constants["n_query
#     # n_query = target_cpu.shape[1] - n_support
#     ####

#     support_idxs = list(map(supp_idxs, classes))    
#     prototypes = select_proto_idx(support_idxs, input_cpu, batch_size)


#     # prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
#     # FIXME when torch will support where as np
#     #### CHANGE: only one query in one particular class
#     q_idxs = list(map(query_idxs, classes))

#     # query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[samples_per_class:], classes))).view(-1)
#     query_samples = select_query_idx(q_idxs, input).cpu()
#     # query_samples = input.to('cpu')[q_idxs]
#     # query_samples = input.to('cpu')[n_support:]

#     dists = euclidean_dist(query_samples.permute(1, 0, 2), prototypes.permute(1, 0, 2))
#     log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, batch_size, -1).permute(2, 0, 1, 3)

#     target_inds = torch.arange(0, n_classes)
#     target_inds = target_inds.view(n_classes, 1, 1)
#     target_inds = target_inds.expand(batch_size, n_classes, n_query, 1).long()

#     ## TODO
#     # print("CHECK THAT DIMENSION OF ACC/LOSS")
#     # loss_val = -log_p_y[0].gather(2, target_inds).squeeze().view(-1).mean()
#     # _, y_hat = log_p_y[0].max(2)
#     # acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

#     loss_val = -log_p_y.gather(3, target_inds).squeeze().view(-1).mean()
#     _, y_hat = log_p_y.max(3)
#     acc_val = y_hat.eq(target_inds.squeeze(3)).float().mean()
#     breakpoint()
    
#     return acc_val, loss_val