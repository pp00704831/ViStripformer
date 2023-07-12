# ViStripformer (Video-Stripformer)
Pytorch Implementation of "ViStripformer: A Token-Efficient Transformer for
Versatile Video Restoration" 

## Installation
```
git clone https://github.com/pp00704831/ViStripformer.git
cd ViStripformer
conda create -n ViStripformer python=3.7
source activate ViStripformer
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python
pip install scipy
```
## Testing
For reproducing our ViStripformer results on GoPro datasets, download "[Video_Stripformer_GoPro](https://drive.google.com/file/d/1MUXhodPZM3OEho2Kmx0vsPCEyVrbfw2f/view?usp=drive_link)"

For reproducing our ViStripformer+ results on GoPro datasets, download "[Video_Stripformer_Plus_GoPro](https://drive.google.com/file/d/1KhZVXVurwefFo_9El1pkhI7_Xvp550yy/view?usp=drive_link)"
