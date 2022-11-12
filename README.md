## [Space-S x KaKR] 그래프 러닝 및 해커톤
### Set environment
```
conda create --name myenv python=3.9
conda activate myenv
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install pyyaml
```
### Run 
```
python train.py
```
### Test
```
python test.py --checkpoint results/best_{epoch}.pt
```
### Result
|Model|MAE|
|:------:|:-----:|
|DimeNet|0.0645|
|SchNet|0.0221|
|TorchMDNet|0.0126|
