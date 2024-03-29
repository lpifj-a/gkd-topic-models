import torch
from torch import nn
from .memory import ContrastMemory
from .memory import ContrastMemory_queue

eps = 1e-7


class CRCDLoss(nn.Module):
    """CRCD Loss function
    
    Args:
        opt.embed_type: fc;nofc;nonlinear
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRCDLoss, self).__init__()
        self.emd_fc_type = opt.embed_type
        print("fc_type: {} ".format(self.emd_fc_type))
        if self.emd_fc_type == "nofc":
            assert opt.s_dim == opt.t_dim
            opt.feat_dim = opt.s_dim
        self.embed_s = Embed(opt.s_dim, opt.feat_dim, self.emd_fc_type)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim, self.emd_fc_type)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        There may be some learnable parameters in embedding layer in teacher side, 
        similar to crd, we also calculate the crcd loss over both the teacher and the student side.
        However, if the emd_fc_type=="nofc", then the 't_loss' term can be omitted.
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """

        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    ''' the contrastive loss is not critical  ''' 
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):    
        bsz = x.shape[0]
        m = x.size(1) - 1

        # 'loss old'
        # noise distribution
        Pn = 1 / float(self.n_data)
        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()
        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        # 'loss new'
        # P_pos = x.select(1, 0) # 64, 1
        # log_P= torch.log(P_pos)
        # # P_neg = x.narrow(1, 1, m) # bs, K, 1
        # # log_N= torch.log(P_pos + P_neg.sum(1))
        # log_N= torch.log(x.sum(1))
        # loss = ((- log_P.sum(0) + log_N.sum(0)) / bsz)

        
        return loss[0]


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128,emd_fc_type='linear'):
        super(Embed, self).__init__()
        if emd_fc_type == "linear":
            self.linear = nn.Linear(dim_in, dim_out)
        elif emd_fc_type == "nonlinear":
            self.linear = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_out),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out, dim_out)
            )
        elif emd_fc_type == "nofc":
            self.linear = nn.Sequential()
        else:
            raise NotImplementedError(emd_fc_type)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x



class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out