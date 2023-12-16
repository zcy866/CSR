import logging
import math
import time

import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils import data
import os
import os.path as osp
import random
import torch.nn.functional as F
import cv2

from augmentation import SoftAugment, RandAugment, RandAugmentMC

logger = logging.getLogger(__name__)

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def load_dataset(args):
    if args.name == 'DomainNet':
        print("**********************")
        print("DomainNet read!")
        print("**********************")
        data_transforms = {
            'src_path': transforms.Compose(
                [transforms.Resize([256, 256]),
                 #transforms.RandomCrop(224),
                 transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]),
            'trg_path': transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip()]),
            'val_path': transforms.Compose(
                [transforms.Resize([224, 224]),
                 #transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]),
            'box_path': transforms.Compose(
                [transforms.Resize([224, 224]),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]),
        }
        train_transform = TransformFixMatch(alpha=1-args.bt)#TransformMPL(args, mean, std, data_transforms['src_path'], if_norm=True )
        train_src_sets = []
        box_src_sets = []
        for src in args.src:
            tset = DomainNetDataset(root=args.data_path, image_list=src+'_train.txt', transform=srcTransform(alpha=1.0), is_fix_trans=True)
            train_src_sets.append(tset)
            box_src_sets.append(DomainNetDataset(root=args.data_path, image_list=src+'_train.txt', transform=data_transforms['box_path']))
        train_tar_set = DomainNetDataset(root=args.data_path, image_list=args.tar[0]+'_train.txt', transform=train_transform, is_fix_trans=True)
        test_tar_set = DomainNetDataset(root=args.data_path, image_list=args.tar[0]+'_test.txt', transform=data_transforms['val_path'])
        box_tar_set = DomainNetDataset(root=args.data_path, image_list=args.tar[0]+'_train.txt', transform=data_transforms['box_path'])
        return train_src_sets, train_tar_set, test_tar_set, train_src_sets + [train_tar_set], box_src_sets, box_tar_set

    '''
    weak_transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         # transforms.RandomCrop(224),
         transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    '''
    train_transform = TransformFixMatch(alpha=1-args.bt)#TransformMPL(args, mean, std, weak_transform )
    test_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    box_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    train_src_sets = []
    box_src_sets = []
    for src in args.src:
        tset = TempImageFolder(root=osp.join(args.data_path, src), transform=srcTransform(alpha=1.0), is_fix_trans=True)
        train_src_sets.append(tset)
        box_src_sets.append(TempImageFolder(root=osp.join(args.data_path, src), transform=box_transform))
    train_tar_set = TempImageFolder(root=osp.join(args.data_path, args.tar[0]), transform=train_transform, is_fix_trans=True)
    test_tar_set = TempImageFolder(root=osp.join(args.data_path, args.tar[0]), transform=test_transform)
    box_tar_set = TempImageFolder(root=osp.join(args.data_path, args.tar[0]), transform=box_transform)
    
    return train_src_sets, train_tar_set, test_tar_set, train_src_sets + [train_tar_set], box_src_sets, box_tar_set

def load_eval_dataset(args):
    if args.name == 'DomainNet':
        data_transforms = {
            'src_path': transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip()]),
            'trg_path': transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip()]),
            'val_path': transforms.Compose(
                [transforms.Resize([224, 224]),
                 #transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]),
            'box_path': transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)]),
        }
        train_transform = TransformMPL(args, mean, std, data_transforms['src_path'], if_norm=True )
        train_src_sets = []
        for src in args.src:
            tset = DomainNetDataset(root=args.data_path, image_list=src+'.txt', transform=data_transforms['val_path'])
            train_src_sets.append(tset)
        train_tar_set = DomainNetDataset(root=args.data_path, image_list=args.tar[0]+'_train.txt', transform=data_transforms['val_path'])
        test_tar_set = DomainNetDataset(root=args.data_path, image_list=args.tar[0]+'_test.txt', transform=data_transforms['val_path'])
        return train_src_sets, train_tar_set, test_tar_set, train_src_sets + [train_tar_set]
    
    weak_transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip()])
    train_transform = TransformMPL(args, mean, std, weak_transform )
    test_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    box_transform = test_transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    train_src_sets = []
    for src in args.src:
        tset = TempImageFolder(root=osp.join(args.data_path, src), transform=test_transform)
        train_src_sets.append(tset)
    train_tar_set = TempImageFolder(root=osp.join(args.data_path, args.tar[0]), transform=test_transform)
    test_tar_set = TempImageFolder(root=osp.join(args.data_path, args.tar[0]), transform=test_transform)
    
    return train_src_sets, train_tar_set, test_tar_set, train_src_sets + [train_tar_set]

class TransformMPL(object):
    def __init__(self, args, mean, std, weak_transform, if_norm=True):
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default

        self.ori = weak_transform
        self.aug = transforms.Compose([
            RandAugment(n=n, m=m)])
            #SoftAugment(n=n, m=m)])
        if if_norm:
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            self.normalize = transforms.Compose([
                transforms.ToTensor()])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(self.ori(x))
        return self.normalize(ori), self.normalize(aug)

class TempImageFolder(datasets.ImageFolder):
    def __init__(self,  root, is_fix_trans=False, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader, is_valid_file=None):
        super(TempImageFolder, self).__init__( root, transform, target_transform,
                 loader, is_valid_file)
        self.imgs = self.samples
        self.is_fix_trans = is_fix_trans
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if self.is_fix_trans:
                sample = self.transform(sample, index)
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

class CCompose(transforms.Compose):
    def __call__(self, x, box, tscale=[[0, 0], [0, 0]]):  # x: [sample, box]
        img = self.transforms[0](x)
        img = self.transforms[1](img, box, tscale)
        for t in self.transforms[2:]:
            img = t(img)
        return img

class srcTransform(object):
    def __init__(self, alpha=0.5):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.all_box = None
        self.weak = CCompose(
            [transforms.Resize([256, 256]),
            twoBoxContrastiveCrop(alpha=alpha, size=224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    def __call__(self, x, index):
        if self.all_box is None:
            weak = self.weak(x, None)
        else:
            wide_box = self.all_box[0][index]
            tar_box = self.all_box[1][index]
            src_box = self.all_box[2][index]

            h0, w0, h1, w1 = tar_box
            th0, tw0, th1, tw1 = src_box
            h_list = [h0, h1, th0, th1]
            w_list = [w0, w1, tw0, tw1]
            h_list.sort()
            w_list.sort()
            intersection_box = [h_list[1], w_list[1], h_list[2], w_list[2]]

            if (h0 == h_list[0] and h1 == h_list[1]) or (h1 == h_list[0] and h0 == h_list[1]) or (th0 == h_list[0] and th1 == h_list[1]) or (th1 == h_list[0] and th0 == h_list[1]):
                intersection_box = [th0, tw0, th1, tw1]
                print("The non-crossing matrix is produced in src")
            elif (w0 == w_list[0] and w1 == w_list[1]) or (w1 == w_list[0] and w0 == w_list[1]) or (tw0 == w_list[0] and tw1 == w_list[1]) or (tw1 == w_list[0] and tw0 == w_list[1]):
                intersection_box = [th0, tw0, th1, tw1]
                print("The non-crossing matrix is produced in src")
            if random.random() > 0.5:
                weak = self.weak(x, [wide_box, intersection_box], [[0, 0], [0, 0]])
            else:
                weak = self.weak(x, None)
        return weak



    def set_box(self, all_box):
        for i in range(0, len(all_box)):
            all_box[i] = all_box[i].cpu()
        self.all_box = all_box

class TransformFixMatch(object):
    def __init__(self, alpha=0.5):
        rgb_mean = (0.485, 0.456, 0.406)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.all_box = None
        self.weak = CCompose([
            transforms.Resize([256, 256]),
            twoBoxContrastiveCrop(alpha=alpha, size=224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self.strong = CCompose([
            transforms.Resize([256, 256]),
            twoBoxContrastiveCrop(alpha=alpha, size=224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            #SoftAugment(n=n, m=m)
            transforms.ToTensor(),
            normalize
        ])

        self.mid = CCompose([
            transforms.Resize([256, 256]),
            twoBoxContrastiveCrop(alpha=alpha, size=224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            #RandAugment(1, 10),
            #SoftAugment(n=n, m=m)
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x, index):
        if self.all_box is None:
            weak = self.weak(x, None)
            strong = self.strong(x, None)
            mid = self.mid(x, None)
        else:
            wide_box = self.all_box[0][index]
            tar_box = self.all_box[1][index]
            r = random.randint(2, len(self.all_box)-1)
            src_box = self.all_box[r][index]

            h0, w0, h1, w1 = tar_box
            th0, tw0, th1, tw1 = src_box
            h_list = [h0, h1, th0, th1]
            w_list = [w0, w1, tw0, tw1]
            h_list.sort()
            w_list.sort()
            intersection_box = [h_list[1], w_list[1], h_list[2], w_list[2]]

            talpha1_h = [1, 1]
            talpha1_w = [1, 1]
            if h0 <= th0:
                talpha1_h[0] = 0
            if h1 >= th1:
                talpha1_h[1] = 0
            if w0 <= tw0:
                talpha1_w[0] = 0
            if w1 >= tw1:
                talpha1_w[1] = 0

            talpha2_h = [1, 1]
            talpha2_w = [1, 1]

            if th0 <= h0:
                talpha2_h[0] = 0
            if th1 >= h1:
                talpha2_h[1] = 0
            if tw0 <= w0:
                talpha2_w[0] = 0
            if tw1 >= w1:
                talpha2_w[1] = 0


            if (h0 == h_list[0] and h1 == h_list[1]) or (h1 == h_list[0] and h0 == h_list[1]) or (th0 == h_list[0] and th1 == h_list[1]) or (th1 == h_list[0] and th0 == h_list[1]):
                talpha1_h = [0, 0]
                talpha1_w = [0, 0]
                talpha2_h = [0, 0]
                talpha2_w = [0, 0]
                intersection_box = [h0, w0, h1, w1]
                print("The non-crossing matrix is produced")
            elif (w0 == w_list[0] and w1 == w_list[1]) or (w1 == w_list[0] and w0 == w_list[1]) or (tw0 == w_list[0] and tw1 == w_list[1]) or (tw1 == w_list[0] and tw0 == w_list[1]):
                talpha1_h = [0, 0]
                talpha1_w = [0, 0]
                talpha2_h = [0, 0]
                talpha2_w = [0, 0]
                intersection_box = [h0, w0, h1, w1]
                print("The non-crossing matrix is produced")
            if random.random() > 0.5:
                if random.random() > 0.5:
                    weak = self.weak(x, [wide_box, intersection_box], [talpha2_h, talpha2_w])
                    strong = self.strong(x, [wide_box, intersection_box], [talpha1_h, talpha1_w])
                    mid = self.mid(x, [wide_box, intersection_box], [[0, 0], [0, 0]])
                else:
                    weak = self.weak(x, [wide_box, intersection_box], [talpha1_h, talpha1_w])
                    strong = self.strong(x, [wide_box, intersection_box], [talpha2_h, talpha2_w])
                    mid = self.mid(x, [wide_box, intersection_box], [[0, 0], [0, 0]])
            else:
                weak = self.weak(x, None)
                strong = self.strong(x, None)
                mid = self.mid(x, None)
        return weak, strong, mid



    def set_box(self, all_box):
        for i in range(0, len(all_box)):
            all_box[i] = all_box[i].cpu()
        self.all_box = all_box


class twoBoxContrastiveCrop(transforms.RandomResizedCrop):  
    def __init__(self, alpha=0.5, is_resized=True, **kwargs):
        super().__init__(**kwargs)
        # a == b == 1.0 is uniform distribution
        self.is_resized = is_resized
        self.alpha = alpha

    def get_params(self, img, bbox, sbox, scale, ratio, tscale):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        # width, height = F._get_image_size(img)
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for i in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = bbox
                th0, tw0, th1, tw1 = sbox

                '''
                if th0 == h0:
                    left_alpha = 1.0 - 0.05
                else:
                    left_alpha = min(self.alpha * abs(th0 - h0) / (h1 - h0) + 1-self.alpha, 0.95)

                if th1 == h1:
                    right_alpha = 1.0 - 0.05
                else:
                    right_alpha = min(self.alpha * abs(h1 - th1) / (h1 - h0) + 1-self.alpha, 0.95)

                if tw0 == w0:
                    up_alpha = 1.0 - 0.05
                else:
                    up_alpha = min(self.alpha * abs(tw0 - w0) / (w1 - w0) + 1-self.alpha, 0.95)

                if tw1 == w1:
                    down_alpha = 1.0 - 0.05
                else:
                    down_alpha = min(self.alpha * abs(w1 - tw1) / (w1 - w0) + 1-self.alpha, 0.95)

                if abs(th0 - h0) <= 0.02 and abs(th1 - h1) <= 0.02:
                    left_alpha = 1-self.alpha
                    right_alpha = 1-self.alpha
                if abs(tw0 - w0) <= 0.02 and abs(tw1 - w1) <= 0.02:
                    up_alpha = 1-self.alpha
                    down_alpha = 1-self.alpha
                '''
                talpha1 = max(self.alpha * (th1 - th0) / (h1 - h0), 0.1)
                talpha2 = max(self.alpha * (tw1 - tw0) / (w1 - w0), 0.1)
                tbeta1 = torch.distributions.beta.Beta(max(talpha1, tscale[0][0]), max(talpha1, tscale[0][1]))
                tbeta2 = torch.distributions.beta.Beta(max(talpha2, tscale[1][0]), max(talpha2, tscale[1][1]))

                #tbeta1 = torch.distributions.beta.Beta(left_alpha, right_alpha)
                #tbeta2 = torch.distributions.beta.Beta(up_alpha, down_alpha)

                ch0 = max(int(height * h0), 0) #avoid a small big-box
                ch1 = min(int(height * (h1+0.07)) - h, height - h)
                cw0 = max(int(width * w0), 0)
                cw1 = min(int(width * (w1+0.07)) - w, width - w)

                ch0 = max(ch0, int(height * th0) - h)
                ch1 = min(ch1, int(height * th1))
                cw0 = max(cw0, int(width * tw0) - w)
                cw1 = min(cw1, int(width * tw1))

                i = min(ch0 + int((ch1 - ch0) * tbeta1.sample()), height - h)
                j = min(cw0 + int((cw1 - cw0) * tbeta2.sample()), width - w)

                return i, j, h, w
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def ori_get_params(self, img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def forward(self, img, box, tscale=[[0,0],[0,0]]):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        if box is None:
            i, j, h, w = self.ori_get_params(img, self.scale, self.ratio)
        else:
            i, j, h, w = self.get_params(img, box[0], box[1], self.scale, self.ratio, tscale)
        if self.is_resized:
            return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        else:
            th = max(h * 224 // 256, 1)
            tw = max(w * 224 // 256, 1)
            ti = random.randint(i, i + h - th)
            tj = random.randint(j, j + w - tw)
            return transforms.functional.crop(img, ti, tj, th, tw)

class DomainNetDataset(data.Dataset):
    def __init__(self, root, is_fix_trans=False, image_list = '', transform = None):
        self.root = root
        self.image_list = image_list
        self.transform = transform
        self.is_fix_trans = is_fix_trans

        self.img_ids = [l.strip().split(' ')[0] for l in open(osp.join(self.root, 'list', self.image_list))]
        self.img_labels = [int(l.strip().split(' ')[1]) for l in open(osp.join(self.root, 'list', self.image_list))]
        self.num_classes = len(np.unique(self.img_labels))
        
    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):

        name = self.img_ids[index]

        image = Image.open(osp.join(self.root, name)).convert('RGB')

        label = self.img_labels[index]

        if self.transform:
            if self.is_fix_trans:
                image = self.transform(image, index)
            else:
                image = self.transform(image)

        return image, label, index

