import logging
import os
import shutil
from collections import OrderedDict
import math
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

class ContinuousDataloader():
    def __init__(self, dataset, args, is_tar=False):
        
        #train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        if is_tar and args.name == 'DomainNet':
            bs = args.batch_size * 2
            print("DomainNet:"+str(bs))
        else:
            bs = args.batch_size
        train_sampler = DistributedSampler
        loader = DataLoader(
            dataset,
            sampler=train_sampler(dataset),
            batch_size=bs,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)
        self.dis = args.world_size
        self.data_loader = loader
        self.iter = iter(self.data_loader)
        self.epoch = 0

    def use_next(self):
        try:
            datas, target, index = next(self.iter)
        except StopIteration:
            self.epoch += 1
            self.data_loader.sampler.set_epoch(self.epoch)
            self.iter = iter(self.data_loader)
            datas, target, index = next(self.iter)
        return datas, target, index

class LrScheduler():

    def __init__(self, optimizer, init_lr, final_iter, warm_steps,args):
        self.init_lr = init_lr
        self.optimizer = optimizer
        self.iter_num = 0
        self.final_iter = final_iter
        self.warm_steps = warm_steps
        self.atten_warm_step = 1000
        self.args = args

    def get_lr(self) -> float:
        if self.iter_num < self.warm_steps:
            lr =  self.init_lr * float(self.iter_num) / float(max(1, self.warm_steps))
            return lr
        lr = self.init_lr / math.pow((1 + 10 * (self.iter_num - self.warm_steps) / (self.final_iter - self.warm_steps)), 0.75)
        return lr
    
    
    def set_lr(self, new_lr):
        self.init_lr = new_lr 
    
    def set_step(self, step):
        self.iter_num = step

    def step(self, iter_num):
        self.iter_num = iter_num
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if "lr_mult" in param_group:
                param_group["lr"] = lr * param_group["lr_mult"]
            else:
                raise("no lr_mult in params group -- utils.py")
        



def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def create_loss_fn(args):
    # if args.label_smoothing > 0:
    #     criterion = SmoothCrossEntropyV2(alpha=args.label_smoothing)
    # else:  
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion.cuda()#.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)


def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.save_path, exist_ok=True)
    if finetune:
        name = f'{args.name}_finetune'
    else:
        name = args.name
    filename = f'{args.save_path}/{name}_last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.save_path}/{args.name}_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def _update(model1, model2, update_fn, cuda=True):
    with torch.no_grad():
        for ema_v, model_v in zip(model1.parameters(), model2.parameters()):
            if cuda:
                model_v = model_v.cuda()#to(device=self.device)
            ema_v.copy_(update_fn(ema_v, model_v))
        for ema_v, model_v in zip(model1.buffers(), model2.buffers()):
            if cuda:
                model_v = model_v.cuda()#to(device=self.device)
            ema_v.copy_(model_v)


#model1 is updated with parameters of model2
def EMA_update(model1, model2, decay, cuda=True):
    update_fn= lambda e, m:decay * e + (1. - decay) * m
    _update(model1, model2, update_fn, cuda)
    

def CopyUpdate(model1, model2, cuda=True):
    with torch.no_grad():
        for ema_v, model_v in zip(model1.parameters(), model2.parameters()):
            if cuda:
                model_v = model_v.cuda()#to(device=self.device)
            ema_v.copy_(model_v)
        
        for ema_v, model_v in zip(model1.buffers(), model2.buffers()):
            if cuda:
                model_v = model_v.cuda()#to(device=self.device)
            ema_v.copy_(model_v)

def CKA(x1, x2):
    t1 = torch.trace(torch.mm(torch.mm(torch.mm(x1,x1.t()),x2),x2.t()))
    t2 = torch.trace(torch.mm(torch.mm(torch.mm(x1,x1.t()),x1),x1.t()))
    t3 = torch.trace(torch.mm(torch.mm(torch.mm(x2,x2.t()),x2),x2.t()))
    return t1/math.sqrt(t2*t3)

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        if self.alpha == 0:
            loss = F.cross_entropy(logits, labels)
        else:
            num_classes = logits.shape[-1]
            alpha_div_k = self.alpha / num_classes
            target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
                (1. - self.alpha) + alpha_div_k
            loss = (-(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()
        return loss

class ClassBasesSelection():
    def __init__(self, args):
        self.branch_num = len(args.src)+1
        self.num_classes = args.num_classes
        self.brach_class_acc = torch.zeros(self.branch_num, self.num_classes).cuda()
        self.thres = args.tar_threshold
        self.decay = 0.9
        
    def split(self, pred, idx):
        pred = pred.detach()
        cutoff = self.thres
        pseudo_label = nn.Softmax(dim=1)(pred)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        cut_thres = cutoff * (1 / (2. - self.brach_class_acc[idx][max_idx]))
        #cut_thres = cutoff * (self.brach_class_acc[idx][max_idx] / (2. - self.brach_class_acc[idx][max_idx]))
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(cut_thres).float()  # convex
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(cutoff).long()
        
        #update
        t_class_acc = torch.zeros(self.num_classes).cuda()
        for i in range(0, len(select)):
            if select[i] == 1:
                t_class_acc[max_idx[i]] += 1
                    
        if sum(select)>0:
            t_class_acc = t_class_acc / max(t_class_acc)
            self.brach_class_acc[idx] = (1-self.decay)*t_class_acc +  self.decay*self.brach_class_acc[idx]
        else:
            self.brach_class_acc[idx] = (1-self.decay)*t_class_acc +  self.decay*self.brach_class_acc[idx]
        
        return mask, select

class SmoothCrossEntropyV2(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, label_smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        assert label_smoothing < 1.0
        self.smoothing = label_smoothing
        self.confidence = 1. - label_smoothing

    def forward(self, x, target):
        if self.smoothing == 0:
            loss = F.cross_entropy(x, target)
        else:
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()
        return loss

class SelfAdaptiveTrainingCE():
    def __init__(self, labels, branch_num, num_classes=65, momentum=0.9):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(branch_num, labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        for i in range(0, branch_num):
            self.soft_labels[i][torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        
        

    def __call__(self, logits, index, idx):
        prob = F.softmax(logits.detach(), dim=1)
        self.soft_labels[idx][index] = self.momentum * self.soft_labels[idx][index] + (1 - self.momentum) * prob 
        # obtain weights
        weights, _ = self.soft_labels[idx][index].max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # compute cross entropy loss, without reduction
        loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[idx][index], dim=1)
        
        # sample weighted mean
        loss = (loss * weights).mean()
        return loss

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
