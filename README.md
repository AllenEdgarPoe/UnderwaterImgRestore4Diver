
## ImgRestoration for Divers!

- This repository is an easy toolkit to restore the underwater image/video using Deep Learning model! 
- It uses the model from the paper **Underwater Ranker: Learn Which is better and How to be better [AAAI 2023 Oral]**


## Get Started

- Install python>=3.6
- Git clone this repository by typing this command in the command line: 
  - `git clone https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver.git`
- Create python virtual environment by using the below command line:
  - `python -m venv venv`
- Activate the virtual environment:
  - `venv\Scripts\activate`
- Install required libraries:
  - `pip install -r requirements.txt`
- Download models from [models](https://drive.google.com/drive/folders/1foW7uXG0GfdGzIDQEhXA82PJryGRQ3aM)
  - get `NU2Net_ckpt.pth` and put it in `./checkpoints/NU2Net_ckpt.pth` 
- Start the program! 
  - `python main.py`


## Functions
- Convert the single image/video file by clicking `Single file` tab 
- Convert the whole image/video files by clicking `Batch` tab. 
- You can set the save directory by designating it through `output_path` tab. If you don't, the default saving path is `./results`
- Comparison results will be saved in `[save_path]/compare`
- Enjoy the beautiful comparison results in the below tab! 

## Examples

### Dialog
<img width="426" alt="image" src="https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/8e61b799-bf4e-4f1b-954c-9c8c5efe37da">

### Reulst (Image comparison)
All of the pictures were taken by me @ Gili.T (Indonesia) 
![KakaoTalk_20230829_131022483_03](https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/e9763721-228c-4d24-bad3-7ec4203aeee4)
![KakaoTalk_20230829_131022483_10](https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/61ee46d6-7519-47f2-a80b-165f77114234)
![KakaoTalk_20230829_131022483](https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/edcbdf16-7cbf-44d5-bf91-5006d5abc678)
![KakaoTalk_20230829_131022483_06](https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/1dfe7bcf-28d6-4931-bbcf-f270e1988d08)
![KakaoTalk_20230829_131022483_04](https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/c2e6db9e-8efb-466d-b5a6-5ccb8b48414d)
![KakaoTalk_20230829_131022483_13](https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/cf5c9891-ce5c-4861-b725-095a8cdf2645)

### Result (Video comparison)

https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/97c73b90-4264-451e-aa4a-e8d7292c8c94



## Acknowledgement

- The code for model architecture is from [UnderwaterRanker](https://github.com/RQ-Wu/UnderwaterRanker)
