from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import os
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetDenseCls
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_PATH = os.path.join(ROOT_DIR, 'shapenetcore_partanno_segmentation_benchmark_v0')

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='C:/Users/Zoe/Desktop/CMPT743A3/codes/seg/seg_model_Chair_no_ft.pth', help='model path')
parser.add_argument('--idx', type=int, default=16, help='model index')
#parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--dataset', type=str, default=DATA_PATH, required=False, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help='class choice')

opt = parser.parse_args()
print(opt)

d = ShapeNetDataset(
    root=opt.dataset,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)

idx = opt.idx

print("model %d/%d" % (idx, len(d)))
point, seg = d[idx]
print(point.size(), seg.size())
point_np = point.numpy()

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy() - 1, :]

state_dict = torch.load(opt.model)
classifier = PointNetDenseCls(k= state_dict['conv2.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
showpoints(point_np, gt, pred_color)
