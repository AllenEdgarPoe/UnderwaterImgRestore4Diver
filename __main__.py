import argparse
import configparser
import sys
import os
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import utils
import cv2
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
from moviepy.editor import AudioFileClip
import time
from PyQt5.QtWidgets import *
from loadingProgressBar import LoadingProgressBar
from PyQt5 import uic


form_class = uic.loadUiType("ui/diver_ui.ui")[0]

class Main(QDialog, form_class):
    def __init__(self, args):
        super().__init__()
        self.intput_path = None
        self.output_path = None
        self.init_ui()
        self.args = args

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        # File browser
        self.select_input_btn = QPushButton('Select File')
        self.select_input_btn.clicked.connect(self.getfiles)
        self.input_label = QLabel()
        self.main_layout.addWidget(self.select_input_btn)
        self.main_layout.addWidget(self.input_label)

        # Save Path
        self.select_output_btn = QPushButton('Select output path')
        self.select_output_btn.clicked.connect(self.getfolder)
        self.output_label = QLabel()
        self.main_layout.addWidget(self.select_output_btn)
        self.main_layout.addWidget(self.output_label)

        # Process Image
        self.process_btn = QPushButton('Process Image')
        self.process_btn.clicked.connect(self.process_inner)
        self.process_label = QLabel()
        self.main_layout.addWidget(self.process_btn)
        self.main_layout.addWidget(self.process_label)

        self.setLayout(self.main_layout)
        self.show()


    #################
    ### functions ###
    #################
    def getfiles(self):
        cwd = os.getcwd()
        fname = QFileDialog.getOpenFileName(self, 'select file', cwd)
        self.input_label.setText(fname[0])

        self.input_path = fname[0]

    def getfolder(self):
        cwd = os.getcwd()
        fname = QFileDialog.getExistingDirectory(self, 'select folder', cwd)
        self.output_label.setText(fname)

        self.output_path = fname
        self.args.save_path = self.output_path

    def define_mode(self, input_path):
        img_extensions = ['.png', '.jpeg', '.jpg']
        video_extensions = ['.mp4', '.avi']
        mode = None
        if os.path.splitext(input_path)[-1] in img_extensions:
            mode = "image"
        elif os.path.splitext(input_path)[-1] in video_extensions:
            mode = "video"

        return mode

    def process_inner(self):
        options = utils.get_option(self.args.opt_path)
        options['model']['resume_ckpt_path'] = self.args.checkpoint_path
        model = utils.build_model(options['model'])
        os.makedirs(self.args.save_path, exist_ok=True)

        mode = self.define_mode(self.input_path)
        if not mode:
            self.process_label.setText('This file extension is not supported')
            return
        else:
            if mode=='image':
                self.process_image(self.input_path, model)
            else:
                self.process_video(self.input_path, model)
            self.process_label.setText(f'Conversion done for file name {self.input_path}')


    def process_image(self, input_path, model):
        bar = LoadingProgressBar()

        lay = QVBoxLayout()
        lay.addWidget(QLabel('Loading...'))
        lay.addWidget(bar)
        mainWidget = QWidget()
        mainWidget.setLayout(lay)
        self.setCentralWidget(mainWidget)

        self.main_layout.addWidget(bar)

        img = Image.open(input_path)
        filename = input_path.split('/')[-1]
        img_w, img_h = img.size[0], img.size[1]
        # upsample = nn.UpsamplingBilinear2d((img_h, img_w))
        img = transforms.Resize((img_h // 32 * 16, img_w // 32 * 16))(img)
        img = transforms.ToTensor()(img).cuda().unsqueeze(0)

        with torch.no_grad():
            result = utils.normalize_img(model(img))
        torchvision.utils.save_image(result, os.path.join(self.args.save_path, filename))
        os.makedirs(os.path.join(self.args.save_path, 'compare'), exist_ok=True)
        torchvision.utils.save_image(torch.cat((img.squeeze(0), result.squeeze(0)), dim=1),
                                     os.path.join(self.args.save_path, 'compare', filename))

    def process_video(self, input_path, model):
        filename = input_path.split('/')[-1]
        vidcap = cv2.VideoCapture(input_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        audioclip = AudioFileClip(input_path)

        new_frames = []
        compare_frames = []
        while (vidcap.isOpened()):
            success, frame = vidcap.read()
            if success:
                img = transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_h, img_w = img.shape[1], img.shape[2]

                img = transforms.Resize((img_h // 32 * 16, img_w // 32 * 16))(img)
                img = img.cuda().unsqueeze(0)

                with torch.no_grad():
                    result = utils.normalize_img(model(img))

                pred = result[0].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1] * 255.0
                pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                new_frames.append(pred)

                # compare frames
                compare = torch.cat((img.squeeze(0), result.squeeze(0)), dim=1).permute(1, 2, 0).detach().cpu().numpy()[
                          :, :, ::-1] * 255.0
                compare = cv2.cvtColor(compare, cv2.COLOR_BGR2RGB)
                compare_frames.append(compare)

            else:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

        try:
            # clip = ImageSequenceClip.ImageSequenceClip(new_frames, fps=fps)
            # clip = clip.set_audio(audioclip)
            # clip.write_videofile(os.path.join(args.save_path, filename), verbose=False, logger=None)

            clip = ImageSequenceClip.ImageSequenceClip(compare_frames, fps=fps)
            clip = clip.set_audio(audioclip)
            clip.write_videofile(os.path.join(self.args.save_path, 'compare', filename), verbose=False, logger=None)
        except Exception as e:
            pass
            # self.process_label.setText(f'{filename} video conversion has not completed due to : {e}')

        vidcap.release()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='convert video or image')
    parser.add_argument('--opt_path', type=str, default='options/NU2Net.yaml')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/NU2Net_ckpt.pth')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--save_path', type=str, default='results')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')
    args.opt_path = config.get("SETTINGS", "OPTION_PATH")
    args.ckpt_path = config.get("SETTINGS", "CHECKPOINT_PATH")
    args.save_path = config.get("SETTINGS", "SAVE_PATH")


    app = QApplication(sys.argv)
    main = Main(args)
    sys.exit(app.exec_())



if __name__ =='__main__':
    run()