import torch
import torch.nn as nn
from torch.nn import functional as F

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss

def dice_loss_all(score, target):
    num_classes = score.shape[1]

    dsc_all = torch.zeros(num_classes)
    for i in range(num_classes):
        dsc_all[i] = dice_loss(score[:, i], target == i)

    loss = torch.mean(dsc_all[1:])
    loss = 1 - loss
    return dsc_all, loss



def kl_divergence(alpha, num_classes, batch):
    ones = torch.ones([batch, num_classes, alpha.shape[-1], alpha.shape[-1]], dtype=torch.float32).cuda()
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl
def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y.float() * (func(S.float()) - func(alpha.float())), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    b = y.shape[0]
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, batch=b)
    return A + kl_div
def edl_digamma_loss(evidence, target, epoch_num, num_classes, annealing_step):
    # evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step
        )
    )
    return loss


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.p = 2

    def forward(self, g_s, g_t):
        loss = sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])
        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()


    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss