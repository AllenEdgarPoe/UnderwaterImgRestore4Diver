
## ImgRestoration for Divers!

- This repository is an easy toolkit to restore the underwater image/video using Deep Learning model! 
- It uses the model from the paper **Underwater Ranker: Learn Which is better and How to be better [AAI 2023 Oral]**


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
- Start the program! 
  - `python main.py`


## Functions
- Convert the single image/video file by clicking `Single file` tab 
- Convert the whole image/video files by clicking `Batch` tab. 
- You can set the save directory by designating it through `output_path` tab. If you don't, the default saving path is `./results`
- Enjoy the beautiful comparison results in the below tab! 

## Examples

<img width="426" alt="image" src="https://github.com/AllenEdgarPoe/UnderwaterImgRestore4Diver/assets/43398106/8e61b799-bf4e-4f1b-954c-9c8c5efe37da">

## Acknowledgement

- The code for model architecture is from [UnderwaterRanker](https://github.com/RQ-Wu/UnderwaterRanker)