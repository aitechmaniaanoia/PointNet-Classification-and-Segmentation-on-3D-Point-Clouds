from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

parser = argparse.ArgumentParser()
# without feature transformation
parser.add_argument('--model', type=str, default = 'C:/Users/Zoe/Desktop/CMPT743A3/codes/cls_no_ft/cls_model_3.pth',  help='model path')

# with feature transformation
parser.add_argument('--model', type=str, default = 'C:/Users/Zoe/Desktop/CMPT743A3/codes/cls/cls_model_3.pth',  help='model path')

parser.add_argument('--num_points', type=int, default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)


## load dataset
test_dataset = ShapeNetDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                               split='test', classification=True, npoints=opt.num_points, data_augmentation=False)
train_dataset = ShapeNetDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                               split='train', classification=True, npoints=opt.num_points, data_augmentation=False)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

classifier = PointNetCls(k=len(test_dataset.classes))
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()


train_correct = 0
total_trainset = 0

# train data
    
for i, data in enumerate(train_dataloader, 0):
    
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('i:%d train accuracy: %f' % (i, correct.item() / 32))
    
    train_correct += correct.item()
    total_trainset += points.size()[0]
    
print("total train accuracy {}".format(train_correct / float(total_trainset)))

    
## test data

total_correct = 0
total_testset = 0
    
for i, data in enumerate(test_dataloader, 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('i:%d test accuracy: %f' % (i, correct.item() /32))
    
    total_correct += correct.item()
    total_testset += points.size()[0]

print("test accuracy {}".format(total_correct / float(total_testset)))
    
    
    
    
    
