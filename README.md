# WiNet: Wavelet-based Incremental Learning for Efficient Medical Image Registration
The official implementation of WiNet (MICCAI2024) [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.13426)  [![MICCAI](https://img.shields.io/badge/MICCAI-68BC71.svg))](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_71)



## Dataset
Thanks [@Junyu](https://github.com/junyuchen245) for [the preprocessed IXI data](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). We followed the same training, validation, and testing protocol.

## Env
```
#pip < 24.1
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```
## Training and Testing
Model weight can be found in
```
WiNet_code/ckpts
```
```
#Training
nohup python tr_IR_3D_WiNet.py >> "./WiNet_diff_IXI.out"
#Testing
python infer_bilinear_WiNet.py
```
The jupyter notebook file is used to get the evaluation results.
```
WiNet_code/Quantitative_Results/analysis.ipynb
```
## Citation
```bibtex
@article{cheng2024winet,
  title={WiNet: Wavelet-based Incremental Learning for Efficient Medical Image Registration},
  author={Cheng, Xinxing and Jia, Xi and Lu, Wenqi and Li, Qiufu and Shen, Linlin and Krull, Alexander and Duan, Jinming},
  journal={arXiv preprint arXiv:2407.13426},
  year={2024}
}
```
```bibtex
@inproceedings{cheng2024winet,
  title={WiNet: Wavelet-based Incremental Learning for Efficient Medical Image Registration},
  author={Cheng, Xinxing and Jia, Xi and Lu, Wenqi and Li, Qiufu and Shen, Linlin and Krull, Alexander and Duan, Jinming},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={761--771},
  year={2024},
  organization={Springer}
}
```
## Acknowledgments
We would like to acknowledge the [Fourier-Net](https://github.com/xi-jia/Fourier-Net) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects, from which we have adopted some of the code used in our work.
