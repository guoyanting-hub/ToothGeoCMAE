Geometry-Guided Multi-View Contrastive Masked Autoencoder for Low-Supervision 3D Tooth Segmentation


Install dependencies:
pip install -r requirements

Compile Chamfer Distance:
cd chamfer_dist

python setup.py install

cd ..

1. Dataset

Teeth3DS (MICCAI 3DTeethSeg’22 Challenge): Contains 1,800 intraoral scans (1,200 training / 600 testing). https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges

3D-IOSSeg: Contains 180 high-resolution scans (120 training / 60 testing). https://www.jianguoyun.com/p/DdSdVsIQivvHDBi03b8FIAA

3. Self-Supervised Pre-training: Train the encoder on unlabeled dental meshes to learn general geometric features
bash pretrain.sh

4. Supervised Fine-tuning: Fine-tune the pre-trained encoder on a limited set of labeled data (e.g., 10%, 30%, or 100%)
bash finetune.sh
