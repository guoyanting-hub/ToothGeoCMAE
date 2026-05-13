# using the modelnet40 as the dataset, and using the processed feature matrixes
import json
import random
from pathlib import Path
import numpy as np
import os
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
from pygem.ffd import FFD
import copy


def randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    # 随机选择xyz三个方向的旋转角度
    angles = [random.choice([0, 30, -30, 15, -15]) for _ in range(3)] # 45, 90, 120, 135, 180, 210, 225, 270, 300, 315
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    mesh1.vertices = rotation.apply(mesh1.vertices)
    return mesh1


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_deformation(mesh: trimesh.Trimesh):
    # ffd = pg.FFD([2, 2, 2])
    ffd = FFD([2, 2, 2])
    random = np.random.rand(6) * 0.1
    ffd.array_mu_x[1, 1, 1] = random[0]
    ffd.array_mu_y[1, 1, 1] = random[1]
    ffd.array_mu_z[1, 1, 1] = random[2]
    ffd.array_mu_x[0, 0, 0] = random[3]
    ffd.array_mu_y[0, 0, 0] = random[4]
    ffd.array_mu_z[0, 0, 0] = random[5]
    vertices = mesh.vertices
    new_vertices = ffd(vertices)
    mesh.vertices = new_vertices
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    vertices = mesh1.vertices - mesh1.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh1.vertices = vertices
    return mesh1


def get_spatial_patch_indices(mesh_path: Path):
    """
    如果离线存在 _patch_idx.npy，就直接读取；否则报错或计算
    """
    patch_file = mesh_path.with_name(mesh_path.stem + "_patch_idx.npy")
    if not patch_file.exists():
        # raise FileNotFoundError(f"Patch indices not found: {patch_file}")
        print(f"Patch indices not found: {patch_file}")
    
    indices = np.load(patch_file)  # shape (250, 64)
    return indices


def generate_face_labels_nearest(mesh, point_labels):   # 根据最近点算法生成面标签
    '''Generate face labels using the nearest point algorithm'''
    face_labels = np.zeros(len(mesh.faces), dtype=int)
    for i, face in enumerate(mesh.faces):
        face_center = mesh.vertices[face].mean(axis=0)
        distances = np.linalg.norm(mesh.vertices - face_center, axis=1)
        nearest_vertex_index = np.argmin(distances)
        face_labels[i] = point_labels[nearest_vertex_index]
    return face_labels


def load_mesh_shape(path, augments=[], request=[], patch_size=64):   # 加载网格数据并进行预处理和特征提取
    mesh = trimesh.load_mesh(path, process=False)   # 加载3D网格模型，process=False 防止 trimesh 自动修改顶点顺序
    # 数据增强：根据augments列表应用随机方向、随机缩放和网格变形
    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)   # 随机旋转
        if method == 'scale':
            mesh = random_scale(mesh)   # 随机缩放
        if method == 'deformation':
            mesh = mesh_deformation(mesh)    # 随机变形
    # 确保只取顶点坐标前三列，忽略颜色信息
    if mesh.vertices.shape[1] > 3:
        mesh.vertices = mesh.vertices[:, :3]
    # 使用 OBJ 中提供的法向量，如果没有则使用 trimesh 自动计算
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and mesh.vertex_normals.shape[0] == mesh.vertices.shape[0]:
        vertex_normals = mesh.vertex_normals
    else:
        vertex_normals = mesh.vertex_normals   # trimesh 会计算
    # 面信息和顶点信息
    F = mesh.faces   # (16000, 3)
    V = mesh.vertices   # (8066, 3)
    Fs = F.shape[0]
    # 计算面顶点、面中心和面法向量
    face_coordinate = V[F.flatten()].reshape(-1, 9)   # (16000, 9)
    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)   # (16000, 3)
    face_normals = mesh.face_normals   # (16000, 3)
    # 计算面曲率近似
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])   # (3, 16000)
    # 13维面特征
    feats = []   # 根据request列表将上述特征添加到 feats 中
    if 'area' in request:
        feats.append(mesh.area_faces)   # 面面积，1维
    if 'normal' in request:
        feats.append(face_normals.T)   # 面法向量，3维
    if 'center' in request:
        feats.append(face_center.T)   # 面中心点，3维
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)   # 面三个内角，3维
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))   # 面曲率，3维
    feats = np.vstack(feats)   # (13, 16000)

    patches_num = 250
    faces_num = patch_size * patches_num   # 16000 = 64 * 250
    # 如果原始面数不足 16000
    if Fs != faces_num:
        raise ValueError(f"Invalid face number: {Fs}")

    # 读取patch划分索引
    indices = get_spatial_patch_indices(Path(path))  # 读取离线 patch 文件
    # 按索引划分patch特征向量
    faces_patch = F[indices]    # (250, 64, 3)，每个patch包含的64个面的顶点索引
    feats_patch = feats[:, indices]    # (13, 250, 64)，每个patch对应的64个面的13维特征
    centers_patch = face_center[indices]    # (250, 64, 3)
    cordinates_patch = face_coordinate[indices]    # (250, 64, 9)

    faces_patcha = np.concatenate((faces_patch, np.zeros((0, 64, 3), dtype=np.float32)), 0)    # (250, 64, 3)
    feats_patcha = np.concatenate((feats_patch, np.zeros((13, 0, 64), dtype=np.float32)), 1)
    centers_patcha = np.concatenate((centers_patch, np.zeros((0, 64, 3), dtype=np.float32)), 0)    # (250, 64, 3)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((0, 64, 9), dtype=np.float32)), 0)    # (250, 64, 9)
    Fs_patcha = np.array(Fs)

    return feats_patcha, centers_patcha, cordinates_patcha, faces_patcha, Fs_patcha


def load_mesh_seg(path, normalize=True, augments=[], request=[], patch_size=64):
    mesh = trimesh.load_mesh(path, process=False)
    base_name = os.path.basename(path)

    label_path = Path(str(path).replace('obj', 'json'))
    with open(label_path) as f:
        segment = json.load(f)
    points_labels = np.array(segment['labels']) #- 1
    faces_labels = generate_face_labels_nearest(mesh, points_labels)    # (16000,)
    # sub_labels = np.array(segment['sub_labels'])

    # 数据增强：随机旋转、随机缩放、归一化（微调）
    for method in augments:
        if method == 'orient':
            mesh = randomize_mesh_orientation(mesh)
        if method == 'scale':
            mesh = random_scale(mesh)
        if method == 'deformation':
            mesh = mesh_deformation(mesh)
    if normalize:
        mesh = mesh_normalize(mesh)
    # 面信息和顶点信息
    F = mesh.faces    # (16000, 3)
    V = mesh.vertices    # (8065, 3)
    Fs = mesh.faces.shape[0]    # 16000
    # 计算面顶点、面中心、顶点法向量和面法向量
    face_coordinate = V[F.flatten()].reshape(-1, 9)    # (16000, 9)
    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)    # (16000, 3)
    vertex_normals = mesh.vertex_normals    # (8065, 3)
    face_normals = mesh.face_normals    # (16000, 3)
    # 计算面曲率信息
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])     # (3, 16000)
    # 13维特征向量
    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))
    feats = np.vstack(feats)    # (13, 16000)

    face_adj = [[] for _ in range(Fs)]
    for i, j in mesh.face_adjacency:
        face_adj[i].append(j)
        face_adj[j].append(i)
    face_adj_fixed = np.full((Fs, 3), -1, dtype=int)  # -1 填充
    for i, neighbors in enumerate(face_adj):
        for j, n in enumerate(neighbors):
            if j < 3:  # 最多3个邻居
                face_adj_fixed[i, j] = n

    # 如果原始面数不足16000
    if Fs != 16000:
        raise ValueError(f"Invalid face number: {Fs}")  
    patches_num = 16000 // patch_size    # 250

    # 读取patch划分索引
    indices = get_spatial_patch_indices(Path(path))  # 读取离线 patch 文件

    # 按索引划分patch特征向量
    faces_patch = F[indices]    # (250, 64, 3)，每个patch包含的64个面的顶点索引
    feats_patch = feats[:, indices]    # (13, 250, 64)，每个patch对应的64个面的13维特征
    centers_patch = face_center[indices]    # (250, 64, 3)
    cordinates_patch = face_coordinate[indices]    # (250, 64, 9)
    label_patch = faces_labels[indices]    # 面标签 (250, 64)

    faces_patcha = np.concatenate((faces_patch, np.zeros((0, 64, 3), dtype=np.float32)), 0)    # (250, 64, 3)
    feats_patcha = np.concatenate((feats_patch, np.zeros((13, 0, 64), dtype=np.float32)), 1)
    feats_patcha = feats_patcha.transpose(1, 2, 0)    # (13, 250, 64)->(250, 64, 13)
    centers_patcha = np.concatenate((centers_patch, np.zeros((0, 64, 3), dtype=np.float32)), 0)    # (250, 64, 3)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((0, 64, 9), dtype=np.float32)), 0)    # (250, 64, 9)

    label_patcha = np.concatenate((label_patch, np.zeros((0, 64), dtype=np.float32)), 0)    # int64->float64
    label_patcha = np.expand_dims(label_patcha, axis=2)    # (250, 64)->(250, 64, 1)
    label_patcha[label_patcha < 0] = 0

    Fs_patcha = np.array(Fs)    # size=1  16000
    Fs_patcha = Fs_patcha.repeat(patches_num * 64).reshape(patches_num, 64, 1)    # (250, 64, 1)

    return faces_patcha, feats_patcha, Fs_patcha, centers_patcha, cordinates_patcha, label_patcha, face_adj_fixed


class Teeth3DSDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augment=None, patch_size=64):
        super().__init__()

        self.dataroot = Path(dataroot)
        self.augments = []
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'normal', 'center']
        self.mesh_paths = []
        self.labels = []
        self.browse_dataroot()
        if train and augment:
            self.augments = augment
        self.patch_size=patch_size
    
    def browse_dataroot(self):
        self.shape_classes = [x.name for x in self.dataroot.iterdir() if x.is_dir()]

        for obj_path in (self.dataroot / self.mode).iterdir():
            if obj_path.is_file() and obj_path.suffix == '.obj':
                self.mesh_paths.append(obj_path)
        self.mesh_paths = np.array(self.mesh_paths)      

    def __getitem__(self, idx):   # 返回双视角数据信息
        if self.mode == 'train':
            feats1, center1, cordinates1, faces1, Fs1 = load_mesh_shape(self.mesh_paths[idx], augments=self.augments, request=self.feats, patch_size=self.patch_size)
            feats2, center2, cordinates2, faces2, Fs2 = load_mesh_shape(self.mesh_paths[idx], augments=self.augments, request=self.feats, patch_size=self.patch_size)
            return feats1, center1, cordinates1, faces1, Fs1, str(self.mesh_paths[idx]), feats2, center2, cordinates2, faces2, Fs2
        else:
            feats1, center1, cordinates1, faces1, Fs1 = load_mesh_shape(self.mesh_paths[idx], augments=self.augments, request=self.feats, patch_size=self.patch_size)
            feats2, center2, cordinates2, faces2, Fs2 = load_mesh_shape(self.mesh_paths[idx], augments=self.augments, request=self.feats, patch_size=self.patch_size)
            return feats1, center1, cordinates1, faces1, Fs1, str(self.mesh_paths[idx]), feats2, center2, cordinates2, faces2, Fs2

    def __len__(self):
        return len(self.mesh_paths)   # 预训练的总样本数


class SegmentationDataset(data.Dataset):
    def __init__(self, dataroot, train=True, augments=None):
        super().__init__()

        self.dataroot = dataroot
        self.augments = []
        self.augments = augments
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curvs', 'normal', 'center']   #center

        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()

    def browse_dataroot(self):

        dataroot = Path(self.dataroot)
        subset_root = dataroot.name    # train/test
        if subset_root == 'train':
            list_txt = dataroot.parent / "finetune_120.txt"
            assert list_txt.exists(), f"{list_txt} not found"
            with open(list_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()   # 兼容 tab / space
                    if len(parts) < 2:
                        continue
                    obj_name = parts[1]    # 01328DDN_lower.obj
                    obj_path = dataroot / obj_name
                    if not obj_path.exists():
                        raise FileNotFoundError(f"OBJ not found: {obj_path}")
                    self.mesh_paths.append(str(obj_path))
            # for obj_path in Path(self.dataroot).iterdir():
            #     if obj_path.is_file() and obj_path.suffix == '.obj':
            #         self.mesh_paths.append(str(obj_path))
        else: 
            for obj_path in Path(self.dataroot).iterdir():
                if obj_path.is_file() and obj_path.suffix == '.obj':
                    self.mesh_paths.append(str(obj_path))
        self.mesh_paths = np.array(self.mesh_paths)

    def __getitem__(self, idx):

        if self.mode == 'train':
            faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha, face_adj = load_mesh_seg(self.mesh_paths[idx], normalize=True, augments=self.augments, request=self.feats)
            return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha, face_adj
        else:
            faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha, face_adj = load_mesh_seg(self.mesh_paths[idx], normalize=True, request=self.feats)
            # print(f'Path:{self.mesh_paths[idx]}')
            return faces_patcha, feats_patcha, Fs_patcha, center_patcha, cordinates_patcha, label_patcha, face_adj
 
    def __len__(self):
        return len(self.mesh_paths)