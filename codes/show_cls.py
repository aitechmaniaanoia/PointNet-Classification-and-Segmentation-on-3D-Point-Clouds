from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetCls
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)


#################################### need to be fixed ######################################
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


for i, data in enumerate(train_dataloader, 0):
    X_points, Y_label = data
    
    X_points = Variable(X_points)
    Y_label = Variable(Y_label)
    
    X_points = X_points.transpose(2, 1)
    
    X_points = X_points.cuda()
    Y_label = Y_label.cuda()
    
    Pred_label, _, _, = classifier(X_points)
    
    # get loss
    loss = F.nll_loss(Pred_label, Y_label)
    
    pred_max = Pred_label.data.max(1)[1]
    acc = pred_max.eq(Y_label.data).cpu().sum()
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), acc / float(32)))
    
    
for i, data in enumerate(test_dataloader, 0):
    X_points, Y_label = data
    
    X_points = Variable(X_points)
    Y_label = Variable(Y_label)
    
    X_points = X_points.transpose(2, 1)
    
    X_points = X_points.cuda()
    Y_label = Y_label.cuda()
    
    Pred_label, _, _, = classifier(X_points)
    
    # get loss
    loss = F.nll_loss(Pred_label, Y_label)
    
    pred_max = Pred_label.data.max(1)[1]
    acc = pred_max.eq(Y_label.data).cpu().sum()
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), acc / float(32)))
    
    
    
    
    
