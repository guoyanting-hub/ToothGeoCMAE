import os
import torch
import sys
import argparse
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import random
from model.dataset_16k_1800_KMeans import SegmentationDataset
from model.meshcmae import Mesh_baseline_seg
import sys
import matplotlib.pyplot as plt
from datetime import datetime

sys.setrecursionlimit(3000)

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


class TrainingVisualizer:
    def __init__(self, log_dir, name):
        self.log_dir = log_dir
        self.name = name
        # 初始化数据列表
        self.epochs = []
        self.losses = []
        self.accuracies = []
        # 设置图表样式
        plt.style.use('seaborn')
        # 创建一个图表和两个Y轴
        self.fig, self.ax1 = plt.subplots(figsize=(12, 6))
        self.ax2 = self.ax1.twinx()
        # 设置图表标题和标签
        self.ax1.set_title(f'Training Metrics - {name}')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss', color='blue')
        self.ax2.set_ylabel('Accuracy', color='red')
        # 设置网格
        self.ax1.grid(True, alpha=0.3)
        
    def update(self, epoch, loss, accuracy):
        # 确保输入是CPU标量
        if torch.is_tensor(loss):
            loss = loss.detach().cpu().item()
        if torch.is_tensor(accuracy):
            accuracy = accuracy.detach().cpu().item()
        # 如果这个epoch的数据已经存在，就更新它
        if epoch in self.epochs:
            idx = self.epochs.index(epoch)
            self.losses[idx] = loss
            self.accuracies[idx] = accuracy
        else:
            # 否则添加新数据
            self.epochs.append(epoch)
            self.losses.append(loss)
            self.accuracies.append(accuracy)
        # 清除当前图表
        self.ax1.clear()
        self.ax2.clear()
        # 重新绘制损失和准确率
        line1 = self.ax1.plot(self.epochs, self.losses, 'b-', label='Loss')
        line2 = self.ax2.plot(self.epochs, self.accuracies, 'r-', label='Accuracy')
        # 更新标签和标题
        self.ax1.set_title(f'Training Metrics - {self.name}')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss', color='blue')
        self.ax2.set_ylabel('Accuracy', color='red')
        # 添加图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        self.ax1.legend(lines, labels, loc='upper right')
        # 设置网格
        self.ax1.grid(True, alpha=0.3)
        # 保存图表
        self.fig.savefig(os.path.join(self.log_dir, 'training_metrics.png'))
        plt.close(self.fig)  # 保存后关闭图表释放内存
    
    def close_figures(self):
        """关闭所有图表以释放内存"""
        plt.close(self.fig)
        plt.close('all')  # 确保关闭所有可能的图表


def setup_logger(args, name, finetune_dataset_len, test_dataset_len):
    # 修改日志目录结构
    log_dir = os.path.join('logs', name)
    os.makedirs(log_dir, exist_ok=True)
    # 使用固定的文件名而不是时间戳
    train_log = os.path.join(log_dir, 'finetune_log.txt')
    # 如果文件不存在，创建并写入表头
    if not os.path.exists(train_log):
        with open(train_log, 'w') as f:
            f.write('Training Configuration:\n')
            for arg, value in vars(args).items():
                f.write(f'{arg}: {value}\n')
            f.write(f'finetune_len: {finetune_dataset_len}\n')
            f.write(f'test_len: {test_dataset_len}\n')
            starting_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(f'[{starting_time}]')
            f.write(f'\n[{starting_time}]\n')
            f.write('Training Progress:\n')
    
    return train_log, log_dir


def compute_cat_iou(pred, target, iou_tabel):  # pred [B,N,C] target [B,N]
    iou_list = []
    target_np = target.cpu().data.numpy()
    pred_np = pred.cpu().data.numpy()

    for j in range(pred.size(0)):
        batch_pred = pred_np[j]  # batch_pred [N,C]
        batch_target = target_np[j]  # batch_target [N]
        batch_choice = np.argmax(batch_pred, axis=1)  # index of max value  batch_choice [N]

        for cat in np.unique(batch_target):
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat, 0] += iou
            iou_tabel[cat, 1] += 1
            iou_list.append(iou)
    return iou_tabel, iou_list


def compute_boundary_mask(labels, face_adj):
    """
    labels: (N,) int
    face_adj: (N, K), K=3, padding 用 -1
    """
    N = labels.shape[0]
    boundary = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in face_adj[i]:
            if j == -1:
                continue        # 忽略padding为-1的邻居
            if labels[j] != labels[i]:
                boundary[i] = True
                break

    return boundary


def boundary_iou_single(pred, gt, face_adj, num_classes):
    """
    pred, gt: (N,) int
    face_adj: 邻接表
    """
    b_iou = np.full(num_classes, np.nan)

    pred_b = compute_boundary_mask(pred, face_adj)
    gt_b = compute_boundary_mask(gt, face_adj)

    for c in range(num_classes):
        pred_c = (pred == c) & pred_b
        gt_c = (gt == c) & gt_b

        inter = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)

        if union > 0:
            b_iou[c] = inter / union

    return b_iou


def test(net, criterion, test_dataset, epoch, args):
    net.eval()
    running_loss = 0.0
    running_corrects = 0
    n_samples = 0
    valid_corrects = 0     # 新增：VALID_CLASSES正确数
    n_valid_faces = 0    # 新增：VALID_CLASSES总face数
    patch_size = 64
    num_of_patch = 0

    # 保存整个测试集预测
    all_preds = []   # list of dicts: {sample_idx, sub_labels}

    VALID_CLASSES = ([0] + list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48)))
    VALID_CLASSES_SET = set(VALID_CLASSES)  # 新增：便于mask
    # 指标统计
    all_shape_ious = []
    iou_tabel = np.zeros((args.seg_parts, 2))
    category_tp = np.zeros(args.seg_parts)
    category_fp = np.zeros(args.seg_parts)
    category_fn = np.zeros(args.seg_parts)
    category_present = np.zeros(args.seg_parts, dtype=bool)
    category_biou_sum = np.zeros(args.seg_parts)
    category_biou_cnt = np.zeros(args.seg_parts)

    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels, face_adj) in enumerate(test_dataset):
        # ---------- 数据搬到 GPU ----------
        face_patch = face_patch.to(device, non_blocking=True)
        feats_patch = feats_patch.to(device, non_blocking=True)
        np_Fs = np_Fs.to(device, non_blocking=True)
        center_patch = center_patch.to(device, non_blocking=True)
        coordinate_patch = coordinate_patch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        faces = face_patch
        feats = feats_patch.float()
        centers = center_patch.float()
        Fs = np_Fs
        coordinates = coordinate_patch.float()

        patch_size = faces.size(2)
        num_of_patch = faces.size(1)

        labels = labels.long().reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]

        # ---------- 前向推理 ----------
        with torch.no_grad():
            outputs, outputs_seg = net(faces, feats, centers, Fs, coordinates)

        outputs = outputs.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        # ---------- loss ----------
        loss_cls = criterion(outputs, labels)
        loss_seg = criterion(outputs_seg, labels)
        loss = args.lw1 * loss_cls + args.lw2 * loss_seg

        _, preds = torch.max(outputs_seg, 1)    # preds [21, 16000]

        running_corrects += torch.sum(preds == labels)
        running_loss += loss.item() * faces.size(0)

        # ================== 新增 VALID_CLASSES accuracy 累加 ==================
        mask_valid = torch.zeros_like(labels, dtype=torch.bool)
        for c in VALID_CLASSES_SET:
            mask_valid |= (labels == c)
        valid_corrects += torch.sum((preds == labels) & mask_valid)
        n_valid_faces += torch.sum(mask_valid)

        # ================= 保存预测 =================
        # patch_num = 250
        label_patcha = preds.reshape(num_of_patch, patch_size, -1)[:num_of_patch, :, 0]
        restored_sub_labels = label_patcha.transpose(1, 0).reshape(-1)
        preds_cpu = restored_sub_labels.cpu().numpy().tolist()

        all_preds.append({
            "sample_idx": i,
            "pred_labels": preds_cpu
        })

        # ================= 指标累积 =================
        outputs_seg_perm = outputs_seg.permute(0, 2, 1)
        iou_tabel, shape_ious = compute_cat_iou(outputs_seg_perm, labels, iou_tabel)
        all_shape_ious.extend(shape_ious)

        preds_np = preds.cpu().numpy()    # (21, 16000)
        labels_np = labels.cpu().numpy()    # (21, 16000)

        for b in range(faces.shape[0]):
            batch_pred = preds_np[b]    # (16000,)
            batch_target = labels_np[b]    

            pred_patch = preds[b].reshape(num_of_patch, patch_size)    # [250, 64]
            gt_patch = labels[b].reshape(num_of_patch, patch_size)
            pred_full = pred_patch.transpose(1, 0).reshape(-1).cpu().numpy()    # (16000,)
            gt_full = gt_patch.transpose(1, 0).reshape(-1).cpu().numpy()
            biou = boundary_iou_single(pred_full, gt_full, face_adj[b], args.seg_parts)    # face_adj[b] [16000, 3]

            for c in range(args.seg_parts):
                tp = np.sum((batch_pred == c) & (batch_target == c))
                fp = np.sum((batch_pred == c) & (batch_target != c))
                fn = np.sum((batch_pred != c) & (batch_target == c))
                category_tp[c] += tp
                category_fp[c] += fp
                category_fn[c] += fn
                if np.sum(batch_target == c) > 0:
                    category_present[c] = True

                if not np.isnan(biou[c]):
                    category_biou_sum[c] += biou[c]
                    category_biou_cnt[c] += 1

    # ================== 计算指标 ==================
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects.double() / (n_samples * num_of_patch * patch_size)
    epoch_valid_acc = valid_corrects.double() / n_valid_faces

    category_iou = np.zeros(args.seg_parts)
    category_sen = np.zeros(args.seg_parts)
    category_ppv = np.zeros(args.seg_parts)
    category_dice = np.zeros(args.seg_parts)
    category_biou = np.zeros(args.seg_parts)

    for c in range(args.seg_parts):
        category_iou[c] = iou_tabel[c, 0] / iou_tabel[c, 1] if iou_tabel[c, 1] > 0 else 0.0

        if category_present[c]:
            category_sen[c] = category_tp[c] / (category_tp[c] + category_fn[c] + 1e-7)
            category_ppv[c] = category_tp[c] / (category_tp[c] + category_fp[c] + 1e-7)
        else:
            category_sen[c] = np.nan
            category_ppv[c] = np.nan
        
        if category_biou_cnt[c] > 0:
            category_biou[c] = category_biou_sum[c] / category_biou_cnt[c]
        else:
            category_biou[c] = np.nan

    valid_ious = []
    for c in VALID_CLASSES:
        if iou_tabel[c, 1] > 0:   # 该类在测试集中出现
            valid_ious.append(category_iou[c])

    for c in VALID_CLASSES:
        denom = 2*category_tp[c] + category_fp[c] + category_fn[c]
        if denom > 0:
            category_dice[c] = 2*category_tp[c] / (denom + 1e-7)
        else:
            category_dice[c] = np.nan

    mean_iou = np.mean(valid_ious)
    mean_dice = np.nanmean(category_dice[VALID_CLASSES])
    mean_sen = np.nanmean([category_sen[c] for c in VALID_CLASSES])
    mean_ppv = np.nanmean([category_ppv[c] for c in VALID_CLASSES])
    mean_biou = np.nanmean([category_biou[c] for c in VALID_CLASSES])

    # ================== 保存最优模型 ==================
    if test.best_acc < epoch_acc:
        test.best_acc = epoch_acc
        torch.save(copy.deepcopy(net.state_dict()),
                   os.path.join('checkpoints', run_name, 'best.pkl'))

    # ================== 控制台打印 ==================
    print(f'Epoch {epoch} | Loss {epoch_loss:.4f} | Acc {epoch_acc:.4f} | Valid Acc {epoch_valid_acc:.4f} | Best {test.best_acc:.4f}')
    print(f'mIoU {mean_iou:.4f} | mBIoU {mean_biou:.4f} | mDice {mean_dice:.4f} | mSEN {mean_sen:.4f} | mPPV {mean_ppv:.4f}')

    # ================== 保存日志 ==================
    # testing_log.txt
    os.makedirs(os.path.join('logs', run_name), exist_ok=True)
    with open(os.path.join('logs', run_name, 'testing_log.txt'), 'a') as f:
        f.write(f'epoch: {epoch} test Loss: {epoch_loss:.4f} Acc {epoch_acc:.4f} Valid Acc: {epoch_valid_acc:.4f} Best: {test.best_acc:.4f}\n')

    # finetune_log.txt
    with open(train.log_path, 'a') as f:
        f.write(f'\nEpoch {epoch} Test Results:\n')
        f.write(f'Test Loss: {epoch_loss:.4f}\n')
        f.write(f'Test Accuracy: {epoch_acc:.4f}\n')
        f.write(f'Test Accuracy (valid): {epoch_valid_acc:.4f}\n')
        f.write(f'Mean IoU: {mean_iou:.4f}\n')
        f.write(f'Mean Boundary IoU: {mean_biou:.4f}\n')
        f.write(f'Mean Dice: {mean_dice:.4f}\n')
        f.write(f'Mean Sensitivity: {mean_sen:.4f}\n')
        f.write(f'Mean PPV: {mean_ppv:.4f}\n')
        f.write('Category IoU:\n')
        for c in VALID_CLASSES:
            f.write(f"Category {c}: {category_iou[c]:.4f}\n")
        f.write('====================\n\n')

        
def train(net, optim, criterion, finetune_dataset, epoch, args):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    valid_corrects = 0     # 新增：VALID_CLASSES正确数
    n_valid_faces = 0    # 新增：VALID_CLASSES总face数
    patch_size = 64
    num_of_patch = 0

    VALID_CLASSES = ([0] + list(range(11, 18)) + list(range(21, 28)) + list(range(31, 38)) + list(range(41, 48)))
    VALID_CLASSES_SET = set(VALID_CLASSES)  # 新增：便于mask

    # 获取当前学习率
    current_lr = optim.param_groups[0]['lr']
    for i, (faces_patch, feats_patch, np_Fs, centers_patch, coordinates_patch, labels, face_adj) in enumerate(finetune_dataset):
        # Prefetch data to GPU
        faces_patch, feats_patch, np_Fs, centers_patch, coordinates_patch, labels = \
            faces_patch.cuda(non_blocking=True), feats_patch.cuda(non_blocking=True), np_Fs.cuda(non_blocking=True), \
            centers_patch.cuda(non_blocking=True), coordinates_patch.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optim.zero_grad()
        faces = faces_patch.cuda()
        patch_size = faces.size(2)
        num_of_patch = faces.size(1)

        feats = feats_patch.to(torch.float32).cuda()
        centers = centers_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinates_patch.to(torch.float32).cuda()
        labels = labels.to(torch.long).cuda()

        labels = labels.reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]
        
        outputs, outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs = outputs.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        loss = criterion(outputs, labels)
        loss_seg = criterion(outputs_seg, labels)
        loss = args.lw1 * loss + args.lw2 * loss_seg
        loss.backward()
        optim.step()
        
        DT, preds = torch.max(outputs_seg, 1)
        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * faces.size(0)

        # ================== 新增 VALID_CLASSES accuracy ==================
        mask_valid = torch.zeros_like(labels, dtype=torch.bool)
        for c in VALID_CLASSES_SET:
            mask_valid |= (labels == c)
        valid_corrects += torch.sum((preds == labels.data) & mask_valid)
        n_valid_faces += torch.sum(mask_valid)
        
        # 每个迭代都输出日志
        print(f"Epoch {epoch}, Iter {i}: Loss = {loss.item():.4f}")
        # 记录每个迭代的训练信息
        if hasattr(train, 'log_path'):
            with open(train.log_path, 'a') as f:
                f.write(f'Epoch {epoch}, Iter {i}: Loss = {loss.item():.4f}\n')
        # 清除不需要的张量以释放内存
        del outputs, outputs_seg, loss, loss_seg
        torch.cuda.empty_cache()
        
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / (n_samples * num_of_patch * patch_size)
    epoch_acc_valid = valid_corrects.double() / n_valid_faces

    print('Epoch: {} Finetune Loss: {:.4f} Acc: {:.4f} Valid Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc, epoch_acc_valid))
    if hasattr(train, 'log_path'):
        with open(train.log_path, 'a') as f:
            f.write('Epoch: {} Finetune Loss: {:.4f} Acc: {:.4f} Valid Acc: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc, epoch_acc_valid))
    # 更新可视化
    if hasattr(train, 'visualizer'):
        train.visualizer.update(epoch, epoch_loss, epoch_acc_valid)
    
    return epoch_loss, epoch_acc, epoch_acc_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--name', type=str, default='Finetune')
    parser.add_argument('--finetune_dataroot', type=str, required=True)
    parser.add_argument('--test_dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--n_epoch', type=int, default=301)
    parser.add_argument('--max_epoch', type=int, default=301)
    parser.add_argument('--lr', type=float, default=1e-4)   # 学习率
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--lr_milestones', type=int, default=None, nargs='+')
    parser.add_argument('--mask_ratio', type=float, default=0.25)   #掩码率
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adamw')
    parser.add_argument('--augment_scale', action='store_true')   # 数据增强
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--channels', type=int, default=13)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--drop_path', type=float, default=0.4)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=16)
    parser.add_argument('--lw1', type=float, default=2)
    parser.add_argument('--lw2', type=float, default=2)
    parser.add_argument('--fpn', action='store_true')
    parser.add_argument('--face_pos', action='store_true')
    parser.add_argument('--seg_parts', type=int, default=50)
    args = parser.parse_args()
    set_seed(args.seed)
    mode = args.mode
    name = args.name
    finetune_dataroot = args.finetune_dataroot
    test_dataroot = args.test_dataroot

    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')
    print(augments)

    # ========== Dataset ==========
    finetune_dataset = SegmentationDataset(finetune_dataroot, train=True, augments=augments)
    finetune_data_loader = data.DataLoader(finetune_dataset, num_workers=args.n_worker, batch_size=args.batch_size, shuffle=True)
    finetune_len = len(finetune_dataset)
    print(f"finetune_len: {finetune_len}")
    test_dataset = SegmentationDataset(test_dataroot, train=False)    # mode='test'
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size, shuffle=False)
    test_len = len(test_dataset)
    print(f"test_len: {test_len}")

    # ========== Network ==========
    net = Mesh_baseline_seg(masking_ratio=args.mask_ratio,
                            channels=args.channels,
                            num_heads=args.heads,
                            encoder_depth=args.encoder_depth,
                            embed_dim=args.dim,
                            decoder_num_heads=args.decoder_num_heads,
                            decoder_depth=args.decoder_depth,
                            decoder_embed_dim=args.decoder_dim,
                            patch_size=args.patch_size,
                            drop_path=args.drop_path,
                            fpn=args.fpn,
                            face_pos=args.face_pos,
                            seg_part=args.seg_parts)

    # 确保使用正确的GPU设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"使用GPU设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    net.to(device)

    # ========== Optimizer ==========
    if args.optim == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones is not None:
        scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.gamma)
    else:
        scheduler = CosineAnnealingLR(optim, T_max=args.max_epoch, eta_min=args.lr_min, last_epoch=-1)

    criterion = nn.CrossEntropyLoss()

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{run_time}_{args.name}"
    checkpoint_path = os.path.join('checkpoints', run_name)
    #checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')
    os.makedirs(checkpoint_path, exist_ok=True)
    if args.checkpoint is not None:
        print('loading checkpoint', args.checkpoint)
        missing_keys, unexpected_keys = net.load_state_dict(torch.load(args.checkpoint), strict=False)
        print("Missing keys (randomly initialized):", missing_keys)
    else:
        for pname, param in net.named_parameters():
            print(f"learn:{pname}")
    # 设置日志文件和可视化器
    train.log_path, log_dir = setup_logger(args, run_name, finetune_len, test_len)
    train.visualizer = TrainingVisualizer(log_dir, run_name)

    train.step = 0
    test.best_acc = 0
    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch()
            print('epoch:', epoch)
            train(net, optim, criterion, finetune_data_loader, epoch, args)
            print('finetune finished')
            if epoch % 10 == 0:
                test(net, criterion, test_data_loader, epoch, args)
                print('test finished')
            scheduler.step()
            print(optim.param_groups[0]['lr'])

    ending_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'[{ending_time}]')
    # 将训练结束时间写入 finetune_log.txt
    with open(train.log_path, 'a') as f:
        f.write(f'[{ending_time}]\n')