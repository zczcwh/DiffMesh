
# DiffMesh
=======

[![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/pdf/2303.13397) <a href="https://zczcwh.github.io/diffmesh_page/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> 

This is the official PyTorch implementation of the paper *"[DiffMesh: A Motion-aware Diffusion Framework for Human Mesh Recovery from Videos](https://arxiv.org/pdf/2303.13397.pdf)"*.



## Installation
Please follow [MotionBert](https://github.com/Walter0807/MotionBERT) to prepare env and data. 

```bash
conda create -n diffmesh python=3.7 anaconda
conda activate diffmesh
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Testing 

To evaluate our model on 3DPW dataset:
``` bash
python train_mesh.py \
--config configs/mesh/DM_ft_pw3d.yaml \
--evaluate checkpoint/mesh/FT_DM_release_DM_ft_pw3d/best_epoch.bin 
```

For HybrIK refinement results
``` bash
python train_mesh_refine.py \
--config configs/mesh/DM_ft_pw3d_refine_hybrik.yaml \
--evaluate checkpoint/mesh/FT_DM_release_DM_ft_pw3d_refine_hybrik/best_epoch.bin
```


## Training

To train on 3DPW dataset: 
``` bash
python train_mesh.py --config configs/mesh/DM_ft_pw3d.yaml --pretrained checkpoint/pretrain/MB_release --checkpoint checkpoint/mesh/FT_DM_release_DM_ft_pw3d
```

If you want to apply   HybrIK refinement
``` bash
python train_mesh_refine.py \
--config configs/mesh/DM_ft_pw3d_refine_hybrik.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/mesh/FT_DM_release_DM_ft_pw3d_refine_hybrik
```

## Inference
Please follow [MotionBert](https://github.com/Walter0807/MotionBERT) to prepare the files required for inference. 
Then run:
``` bash
python infer_wild_mesh.py --vid_path example/a4/a4.mp4 --json_path example/a4/alphapose-a4.json --out_path example/a4/output --clip_len 16
```

## Citation

If you find our work useful for your project, please consider citing the paper:

```bibtex

@inproceedings{zheng2025diffmesh,
  title={DiffMesh: A Motion-aware Diffusion Framework for Human Mesh Recovery from Videos},
  author={Zheng, Ce and Liu, Xianpeng and Peng, Qucheng and Wu, Tianfu and Wang, Pu and Chen, Chen},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) },
  year={2025}
}
```

