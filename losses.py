import torch
import torch.nn.functional as F
from dexi_utils import *

def PDNet_bce_loss(inputs, targets):
    # mask = targets.float()
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(reduction='none')(inputs, targets.float())
    final_cost = torch.sum(cost)
    return final_cost

def hed_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.1).float()).float()
    num_negative = torch.sum((mask <= 0.).float()).float()

    mask[mask > 0.1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), targets.float())

    return l_weight*torch.sum(cost)


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    # reduction="none"会返回batch中每个loss， dim=input, [batch, c, h,w]
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost

def bdcn_lossORI(inputs, targets, l_weigts=1.1,cuda=False):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    # print(cuda)
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * 1.1 / valid  # balance = 1.1
    weights = torch.Tensor(weights)
    # if cuda:
    weights = weights.cuda()
    inputs = torch.sigmoid(inputs)
    loss = torch.nn.BCELoss(weights, reduction='sum')(inputs.float(), targets.float())
    return l_weigts*loss

def rcf_loss(inputs, label):

    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return 1.*torch.sum(cost)

# ------------ cats losses ----------

def bdrloss(prediction, label, radius,device='cpu'):
    '''
    The boundary tracing loss that handles the confusing pixels.
    '''

    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)     # 4维 [batch, c, h w]
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label       # pred上 只留下label=1的
    # 用1filter对进行卷积，像素位置上是对patch内的值的求和。
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)    # 对卷积完的，再次只留下label=1的
    # texture_mask： 点为中心的patch内有无edge。 只要label对应区域内有》0的值，即edge，texture_mask对应位置就》0
    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    # 不计算patch中的edge。 论文公式2的 R \ L 部分
    mask[label == 1] = 0        
    # 这里乘(1-label) 和上一行作用重复了
    pred_texture_sum = F.conv2d(prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)       # 对预测错的

    softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()



def textureloss(prediction, label, mask_radius, device='cpu'):
    '''
    The texture suppression loss that smooths the texture regions.
    '''
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None, stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)


def cats_loss(prediction, label, l_weight=[0.,0.], device='cpu'):
    # tracingLoss
    tex_factor,bdr_factor = l_weight
    balanced_w = 1.1
    label = label.float()
    prediction = prediction.float()
    with torch.no_grad():
        mask = label.clone()        # 原来是0,1值

        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        beta = num_negative / (num_positive + num_negative)
        # mask[mask == 1] = beta      # 给positive赋权重beta
        # mask[mask == 0] = balanced_w * (1 - beta)       # 这里感觉写反了
        posit_mask = mask == 1
        mask[mask == 1] = balanced_w * beta
        mask[mask == 0] = 1 - beta
        mask[mask == 2] = 0     # 一般没有2
    prediction = torch.sigmoid(prediction)
    # print('bce')
    ## 给hard sample部分的loss加倍
    # c = torch.nn.functional.binary_cross_entropy(
    #     prediction.float(), label.float(), weight=mask, reduce=False)
    # thres = c.min() + 0.8*(c.max() - c.min())
    # hard_mask = (c>thres) * (posit_mask == 1)
    # mask[hard_mask] *= 10       # 正样本的hard sample的loss加倍
    
    cost = torch.sum(torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False))
    label_w = (label != 0).float()
    # print('tex')
    textcost = textureloss(prediction.float(), label_w.float(), mask_radius=4, device=device)
    bdrcost = bdrloss(prediction.float(), label_w.float(), radius=4, device=device)

    return cost + bdr_factor * bdrcost + tex_factor * textcost