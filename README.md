ToothGeoCMAE is a self-supervised framework designed to learn discriminative geometric representations from unlabeled 3D dental meshes. By integrating Masked Autoencoding (MAE) with Multi-view Contrastive Learning, the model achieves state-of-the-art performance in 3D tooth segmentation using only 30% of labeled data.

Dual-Branch Architecture: Jointly optimizes local geometric reconstruction and global multi-view consistency.
GFE Module: A Geometry-guided Face Embedding module using cross-attention to couple local morphology with spatial positions.
Low-Supervision Efficiency: Outperforms existing self-supervised baselines and matches fully supervised methods with minimal annotations.
Mesh-specific Design: Specifically tailored for Intraoral Scans (IOS) with dense geometric continuity and similar adjacent instances.


Install dependencies:
pip install -r requirements

Compile Chamfer Distance:
cd chamfer_dist
python setup.py install
cd ..

1. Dataset
Teeth3DS (MICCAI 3DTeethSeg’22 Challenge): Contains 1,800 intraoral scans (1,200 training / 600 testing). https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges
3D-IOSSeg: Contains 180 high-resolution scans (120 training / 60 testing). https://www.jianguoyun.com/p/DdSdVsIQivvHDBi03b8FIAA

2. Self-Supervised Pre-training (Stage 1)
Train the encoder on unlabeled dental meshes to learn general geometric features:
bash pretrain.sh
Note: This script calls train_pretrain_16k_1800_KMeans.py. You can modify hyperparameters like mask_ratio (default: 0.5) inside the script.

3. Supervised Fine-tuning (Stage 2)
Fine-tune the pre-trained encoder on a limited set of labeled data (e.g., 10%, 30%, or 100%):
bash finetune.sh
Note: This script calls train_finetune_16k_1800_KMeans.py using the checkpoints generated from Stage 1.