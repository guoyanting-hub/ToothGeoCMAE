from collections import Counter
import torch
import os
import sys
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import random
from concurrent.futures import ThreadPoolExecutor
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.dataset import SegmentationDataset
from model.meshmae import Mesh_baseline_seg, Mesh_encoder
from model.reconstruction import save_results
import sys

sys.setrecursionlimit(3000)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha = self.alpha[targets]
            F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def train(net, optim, criterion, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    patch_size = 64
    num_of_patch = 0
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels) in enumerate(
            train_dataset):
        # Prefetch data to GPU
        face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels = \
            face_patch.cuda(non_blocking=True), feats_patch.cuda(non_blocking=True), np_Fs.cuda(non_blocking=True), \
            center_patch.cuda(non_blocking=True), coordinate_patch.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optim.zero_grad()
        faces = face_patch.cuda()
        patch_size = faces.size(2)
        num_of_patch = faces.size(1)

        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        labels = labels.to(torch.long).cuda()

        labels = labels.reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]

        outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        if labels.min()<0:
            print(f"min:{labels.min}")
        if(labels.max()>=args.seg_parts):
            print(f"max:{labels.max()}")
        assert labels.min() >= 0 and labels.max() < args.seg_parts, "Label values are out of range"

        loss = criterion(outputs_seg, labels)

        DT, preds = torch.max(outputs_seg, 1)
        print(f"Dt:{DT}")
        print(f"preds:{preds}")
        print(f"labels:{labels.data}")
        print(f"loss:{loss}")
        running_corrects += torch.sum(preds == labels.data)

        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)
        print(f"iter:{i}, loss:{loss.item()}")
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / (n_samples * num_of_patch * patch_size)
    print('epoch: {:} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    message = 'epoch: {:} Train Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)




def test(net, criterion, test_dataset, epoch, args):
    net.eval()
    acc = 0
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels) in enumerate(
            test_dataset):
        # Prefetch data to GPU
        face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels = \
            face_patch.cuda(non_blocking=True), feats_patch.cuda(non_blocking=True), np_Fs.cuda(non_blocking=True), \
            center_patch.cuda(non_blocking=True), coordinate_patch.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()

        labels = labels.to(torch.long).cuda()
        labels = labels.reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]
        with torch.no_grad():
            outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        loss = criterion(outputs_seg, labels)
        _, preds = torch.max(outputs_seg, 1)

        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * faces.size(0)
        print(f"iter:{i}, loss:{loss.item()}")
    epoch_acc = running_corrects.double() / (n_samples * 16384)
    epoch_loss = running_loss / n_samples

    if test.best_acc < epoch_acc:
        test.best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join('checkpoints', name, 'best.pkl'))
    print('epoch: {:} test Loss: {:.4f} Acc: {:.4f} Best: {:.4f}'.format(epoch, epoch_loss, epoch_acc,test.best_acc))
    message = 'epoch: {:} test Loss: {:.4f} Acc: {:.4f} Best: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc,
                                                                               test.best_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)




if __name__ == '__main__':
    seed_torch(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_milestones', type=int, default=None, nargs='+',)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--drop_path', type=float, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--lw1', type=float, default=0.5)
    parser.add_argument('--lw2', type=float, default=0.5)
    parser.add_argument('--fpn', action='store_true')
    parser.add_argument('--face_pos', action='store_true')
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--dataset_name', type=str, default='alien', choices=['alien', 'human'])
    parser.add_argument('--seg_parts', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    args = parser.parse_args()
    mode = args.mode
    name = args.name
    dataroot = args.dataroot
    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')
    train_dataset = SegmentationDataset(dataroot, train=True, augments=augments)
    test_dataset = SegmentationDataset(dataroot, train=False)

    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True, prefetch_factor=2)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True, prefetch_factor=2)
    # TODO:统计train和test所有的label的class_frequencies并输出
    
    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}")
    # ========== Network ==========

    net = Mesh_encoder(masking_ratio=args.mask_ratio,
                            channels=args.channels,
                            num_heads=args.heads,
                            encoder_depth=args.encoder_depth,
                            embed_dim=args.dim,
                            decoder_num_heads=args.decoder_num_heads,
                            decoder_depth=args.decoder_depth,
                            decoder_embed_dim=args.decoder_dim,
                            patch_size=args.patch_size,
                            drop_path=args.drop_path,
                            #fpn=args.fpn,
                            face_pos=args.face_pos,
                            seg_part=args.seg_parts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # ========== Optimizer ==========
    if args.optim == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones is not None:
        scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.gamma)
    else:

        scheduler = CosineAnnealingLR(optim, T_max=args.max_epoch, eta_min=args.lr_min, last_epoch=-1)
    
    class_frequencies = np.array([1.2276e+06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.9579e+04,
        4.1244e+04, 3.5252e+04, 5.1447e+04, 5.3876e+04, 9.0117e+04, 4.6452e+04,
        4.2700e+02, 0.0000e+00, 0.0000e+00, 6.0716e+04, 4.2130e+04, 3.6838e+04,
        5.0698e+04, 5.4257e+04, 9.0859e+04, 4.5951e+04, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 4.7501e+04, 5.0354e+04, 5.4067e+04, 6.3254e+04, 7.4016e+04,
        1.2183e+05, 6.3640e+04, 7.5600e+02, 0.0000e+00, 0.0000e+00, 4.6837e+04,
        5.0982e+04, 5.6489e+04, 6.6434e+04, 7.3117e+04, 1.1755e+05, 5.7867e+04,
        6.3400e+02, 0.0000e+00])  # 根据类别数量调整权重
    
    num_classes = 50  # 假设有50个类别
    # 计算类别权重时忽略频率为0的类别
    valid_class_frequencies = class_frequencies[class_frequencies > 0]
    valid_class_indices = np.where(class_frequencies > 0)[0]

    # 计算类别权重
    class_weights = np.zeros_like(class_frequencies)
    class_weights[valid_class_indices] = 1.0 / valid_class_frequencies
    class_weights = class_weights / class_weights.sum() * len(valid_class_indices)

    print(class_weights)
    # 将类别权重转换为tensor并移动到GPU
    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    #criterion = FocalLoss(alpha=class_weights)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = os.path.join('checkpoints', name)
    #checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')

    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint), strict=False)

    train.step = 0
    test.best_acc = 0

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch()
            print('epoch:', epoch)
            train(net, optim, criterion, train_data_loader, epoch, args)
            print('train finished')
            test(net, criterion, test_data_loader, epoch, args)
            print('test finished')
            scheduler.step()
            print(optim.param_groups[0]['lr'])


    else:
        test(net, criterion, test_data_loader, 0, args)
