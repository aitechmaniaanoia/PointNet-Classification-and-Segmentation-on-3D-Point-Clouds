from __future__ import print_function
import argparse
import os
import time
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_PATH = os.path.join(ROOT_DIR, 'shapenetcore_partanno_segmentation_benchmark_v0')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
#parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset', type=str, default=DATA_PATH, required=False, help="dataset path")
parser.add_argument('--feature_transform', default = True, action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# dataset = ShapeNetDataset(
#     root=opt.dataset,
#     #root='shapenetcore_partanno_segmentation_benchmark_v0/',
#     classification=True,
#     npoints=opt.num_points)

# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=opt.batchSize,
#     shuffle=True,
#     num_workers=int(opt.workers))

## train dataset
dataset = ShapeNetDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                          split='train', classification=True, npoints=opt.num_points, data_augmentation=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)


## test dataset
test_dataset = ShapeNetDataset(root='shapenetcore_partanno_segmentation_benchmark_v0',
                               split='test', classification=True, 
                               npoints=opt.num_points, data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True)#, num_workers=int(opt.workers))


print(len(dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
num_batch = len(dataset) / opt.batchSize

best_val = 0
start_time = time.time()

for epoch in range(opt.nepoch):
    scheduler.step()
    
    train_correct = 0
    total_trainset = 0
    
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        train_correct += correct.item()
        total_trainset += points.size()[0]
        
        #print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
    
    print('[%d] time: %f' % (epoch, time.time()-start_time))
    print("train accuracy {}".format(train_correct / float(total_trainset)))
    

       # if i % 10 == 0:
            # j, data = next(enumerate(testdataloader, 0))
            # points, target = data
            # target = target[:, 0]
            # points = points.transpose(2, 1)
            # points, target = points.cuda(), target.cuda()
            # classifier = classifier.eval()
            # pred, _, _ = classifier(points)
            # loss = F.nll_loss(pred, target)
            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.data).cpu().sum()
            # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            
    # test 
    total_correct = 0
    total_testset = 0
    
    #for i,data in tqdm(enumerate(testdataloader, 0)):
    for j, data in enumerate(testdataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    
    print("test accuracy {}".format(total_correct / float(total_testset)))
            
    #torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    
    val_acc = total_correct / float(total_testset)
    
    if val_acc > best_val:
        best_val = val_acc
        
        torch.save(classifier.state_dict(), '%s/cls_best_model_ft.pth' %opt.outf)
        
                                    


