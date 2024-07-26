# WiNet: Wavelet-based Incremental Learning for Efficient Medical Image Registration
The official implementation of WiNet (MICCAI2024) [arxiv](https://arxiv.org/abs/2407.13426)



## Dataset
Thanks [@Junyu](https://github.com/junyuchen245) for [the preprocessed IXI data](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md). We followed the same training, validation, and testing protocol.


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


## Acknowledgments
We would like to acknowledge the [Fourier-Net](https://github.com/xi-jia/Fourier-Net) and [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) projects, from which we have adopted some of the code used in our work.
