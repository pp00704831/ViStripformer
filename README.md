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

**For Deblurring** </br>
* You can find the pre-trained weights for deblurring in "[Here](https://drive.google.com/drive/folders/1UDNPTsGrzhW40yqsH6cXBqwRABBv7x2K?usp=drive_link)" </br>
* Download the pre-trained weights for deblurring and put them into './weights/Deblur'
* For reproducing our ViStripformer results on GoPro datasets, download 'Video_Stripformer_GoPro.pth' </br>
* For reproducing our ViStripformer+ results on GoPro datasets, download 'Video_Stripformer_Plus_GoPro.pth' </br>
* For reproducing our ViStripformer results on BSD datasets, download 'Video_Stripformer_BSD.pth' 
