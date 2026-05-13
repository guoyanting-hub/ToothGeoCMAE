import torch
import os
import argparse
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import torch.utils.data as data
from model.dataset_IOSSeg import Teeth3DSDataset
from model.meshcmae import Mesh_cmae
from transformers import get_cosine_schedule_with_warmup
from datetime import datetime
import matplotlib.pyplot as plt
import random
from pathlib import Path
import trimesh

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
        self.losses = []
        self.epochs = []
        self.log_dir = log_dir
        self.name = name
        
        # 设置图表样式
        try:
            # Matplotlib 3.8+ 不再支持 'seaborn' 名称，优先使用兼容样式
            if 'seaborn-v0_8' in plt.style.available:
                plt.style.use('seaborn-v0_8')
            elif 'ggplot' in plt.style.available:
                plt.style.use('ggplot')
            else:
                plt.style.use('default')
        except Exception:
            # 任意异常时回退为默认样式，确保训练不被样式问题阻塞
            plt.style.use('default')

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(f'Training Loss - {name}')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        
    def update(self, epoch, loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # 清除当前图表
        self.ax.clear()
        
        # 重新绘制
        self.ax.plot(self.epochs, self.losses, 'b-', label='Training Loss')
        self.ax.set_title(f'Training Loss - {self.name}')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True)
        self.ax.legend()
        
        # 保存图表
        plt.savefig(os.path.join(self.log_dir, 'training_loss.png'))


def setup_logger(args, time, pretrain_dataset_len):
    # 创建logs目录
    # log_dir = os.path.join('logs', args.name)
    log_dir = os.path.join('logs_IOSSeg', f"{time}_{args.name}")

    # base_dir = '/home/yjy'
    # log_dir = os.path.join(base_dir, 'logs_IOSSeg', f"{run_time}_{args.name}")
    
    os.makedirs(log_dir, exist_ok=True)
    print("日志目录:", log_dir)
    log_file = os.path.join(log_dir, f'training_log.txt')
    
    # 写入训练配置
    with open(log_file, 'w') as f:
        f.write('Training Configuration:\n')
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
        f.write(f'pretrain_len: {pretrain_dataset_len}\n')
        f.write('\nTraining Progress:\n')
    
    return log_file, log_dir


def log_results(log_file, epoch, epoch_loss, is_best=False):
    with open(log_file, 'a') as f:
        log_line = f'Epoch {epoch}  Train Loss: {epoch_loss:.4f}'
        if is_best:
            log_line += ' (Best)'
        f.write(log_line + '\n')


def train(net, optim, scheduler, run_name, train_dataset, epoch, args, log_file, visualizer, checkpoint_path, device='cuda'):
    net.train()   # 切换模式为训练模式
    running_loss = 0
    n_samples = 0

    for it, (feats_patch1, center_patch1, coordinate_patch1, face_patch1, np_Fs1, mesh_paths, feats_patch2, center_patch2, coordinate_patch2, face_patch2, np_Fs2) in enumerate(train_dataset):   # enumerate(train_dataset)=遍历 DataLoader，DataLoader 每次yield一个 batch的样本
    # 每次迭代就是一个batch，调用Dataloader，得到patch个样本的两个视图的数据，用于对比训练
        optim.zero_grad()   # 防止梯度累积

        # 将所有输入统一转成 float32 Tensor 并移动到GPU
        faces1 = face_patch1.to(device=device, dtype=torch.float32)   # 面索引 [24, 250, 64, 3]
        feats1  = feats_patch1.to(device=device, dtype=torch.float32)   # 13维特征：面面积、面法向量、面中心点、面内角、面曲率 [b, c, h, p]=[24, 13, 250, 64]
        centers1 = center_patch1.to(device=device, dtype=torch.float32)   # 面中心点坐标 [24, 250, 64, 3]
        cordinates1 = coordinate_patch1.to(device=device, dtype=torch.float32)   # 面顶点坐标 [24, 250, 64, 9]
        Fs1 = np_Fs1.to(device=device, dtype=torch.float32)   # patch个样本的面数信息 [24]

        faces2 = face_patch2.to(device=device, dtype=torch.float32)
        feats2  = feats_patch2.to(device=device, dtype=torch.float32)
        centers2 = center_patch2.to(device=device, dtype=torch.float32)
        cordinates2 = coordinate_patch2.to(device=device, dtype=torch.float32)
        Fs2 = np_Fs2.to(device=device, dtype=torch.float32)

        n_samples += faces1.shape[0]   # “全局”变量，统计样本数，用于计算每个epoch平均loss
        # print(n_samples)

        loss = net(faces1, feats1, centers1, Fs1, cordinates1, faces2, feats2, centers2, Fs2, cordinates2)   # 前向传播
        loss.backward()   # 反向传播，计算所有可学习参数的梯度
        optim.step()   # 使用Adamw更新模型参数

        running_loss += loss.item() * faces1.size(0)   # 累加计算整个epoch的loss，loss.item()为batch平均loss，faces1.size(0)即为batch size

    epoch_loss = running_loss / n_samples     # epoch平均loss
    scheduler.step()   # 学习率调度。如果用了 MultiStepLR，按 milestone 衰减；否则用 Cosine + Warmup

    # 保存最优模型
    is_best = False
    if train.best_loss > epoch_loss > 0:
        train.best_loss = epoch_loss
        train.best_epoch = epoch
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join(checkpoint_path, f'loss-{epoch_loss:.4f}-{epoch:.4f}.pkl'))   # 保存checkpoint
        is_best = True

    print(f'Epoch {epoch}  Train Loss: {epoch_loss:.4f}')
    log_results(log_file, epoch, epoch_loss, is_best)
    visualizer.update(epoch, epoch_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--name', type=str, default='Pretrain')
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=64)   # 每个patch包含的面数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=400)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--weight', type=float, default=0.5)   # 重建损失权重
    parser.add_argument('--channels', type=int, default=13)   # 输入的面特征维度
    parser.add_argument('--lr_milestones', type=str, default="none")
    parser.add_argument('--num_warmup_steps', type=int, default=2)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--n_dropout', type=int, default=2)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=16)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--n_classes', type=int, default=50)
    parser.add_argument('--no_center_diff', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    args = parser.parse_args()
    mode = args.mode
    dataroot = args.dataroot
    set_seed(args.seed)

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')

    # ========= 在创建Dataset之前，先扫描 =========
    train_dataset = Teeth3DSDataset(dataroot, train=True, augment=augments, patch_size=args.patch_size)    # 得到每个样本的双视图数据
    pretrain_len = len(train_dataset)
    print(f"pretrain_len: {pretrain_len}")
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size, shuffle=True, pin_memory=True, worker_init_fn=worker_init_fn)   # DataLoader会调用Dataset的__getitem__ batch_size次，得到batch_size个样本，然后把它们堆叠起来

    # ========== Network ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Mesh_cmae(masking_ratio=args.mask_ratio,
                   channels=args.channels,
                   num_heads=args.heads,
                   encoder_depth=args.encoder_depth,
                   embed_dim=args.dim,
                   decoder_num_heads=args.decoder_num_heads,
                   decoder_depth=args.decoder_depth,
                   decoder_embed_dim=args.decoder_dim,
                   patch_size=args.patch_size,
                   ).to(device)

    # ========== Optimizer ==========
    if args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones is not None and args.lr_milestones.lower() != 'none':
        ms = args.lr_milestones
        ms = ms.split()
        ms = [int(j) for j in ms]
        scheduler = MultiStepLR(optimizer, milestones=ms, gamma=0.1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.num_warmup_steps),
                                                    num_training_steps=args.max_epoch + 1)
    print(scheduler)

    
    # ==========日志和损失可视化==========
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{run_time}_{args.name}"
    log_file, log_path = setup_logger(args, run_time, pretrain_len)
    visualizer = TrainingVisualizer(log_path, args.name)

    # ========== MISC ==========
    checkpoint_dir = './checkpoints_IOSSeg'
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_time}_{args.name}")

    # base_dir = '/home/yjy'
    # checkpoint_dir = os.path.join(base_dir, 'checkpoints_IOSSeg')
    # checkpoint_path = os.path.join(checkpoint_dir, f"{run_time}_{args.name}")

    os.makedirs(checkpoint_path, exist_ok=True)
    if args.checkpoint is not None and args.checkpoint.lower() != 'none':
        print('loading checkpoint', args.checkpoint)
        net.load_state_dict(torch.load(args.checkpoint), strict=False)

    # ========== Start Training ==========
    train.best_loss = 999
    train.best_epoch = 0
    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            train(net, optimizer, scheduler, run_name, train_data_loader, epoch, args, log_file, visualizer, checkpoint_path)   # 调用train()完成 一个epoch的完整训练，循环训练n_epoch次
