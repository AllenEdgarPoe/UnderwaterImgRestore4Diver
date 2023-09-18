import argparse
import configparser
import sys
import os

import qimage2ndarray as qimage2ndarray
from PIL import Image
import torchvision
from PyQt5.QtGui import QPixmap, QImage
from torchvision import transforms
import torch
import utils
import cv2
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
from moviepy.editor import AudioFileClip
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyQt5.QtCore import QThread, Qt
from loadingProgressBar import LoadingProgressBar
from PyQt5 import uic



form_class = uic.loadUiType("ui/diver_ui.ui")[0]
TIME_LIMIT = 10

class WindowDiver(QMainWindow, form_class):
    def __init__(self, args):
        super().__init__()
        self.setupUi(self)
        self.args = args
        self.inputfolder_path = None
        self.input_path = None
        self.textEdit.setReadOnly(True)
        self.event_loop = QEventLoop()


        # Start event loop

        self.pushButton.clicked.connect(self.getfiles)  # get path for input file
        self.pushButton_2.clicked.connect(self.getoutputfolder)  # get folder path for save path
        self.pushButton_5.clicked.connect(self.getinputfolder)

        self.pushButton_3.clicked.connect(self.process_inner) # process the image

        self.label_3.hide()

        self.show()
        self.event_loop.exec_()



    def getfiles(self):
        cwd = os.path.join(os.getcwd(),'examples')
        fname = QFileDialog.getOpenFileName(self, 'select file', cwd)
        self.label.setText(f'input path: {fname[0]}')
        self.input_path = fname[0]

        self.textEdit.setText('')
        self.label_3.show()

    def getinputfolder(self):
        cwd = os.getcwd()
        fname = QFileDialog.getExistingDirectory(self, 'select folder', cwd)

        self.label_5.setText(f'input folder: {fname}')
        self.textEdit.setText('')
        self.label_5.show()

        self.inputfolder_path = fname


    def getoutputfolder(self):
        cwd = os.getcwd()
        fname = QFileDialog.getExistingDirectory(self, 'select folder', cwd)

        self.label_2.setText(f'output folder: {fname}')
        self.label_2.show()

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
        try:
            options = utils.get_option(self.args.opt_path)
            options['model']['resume_ckpt_path'] = self.args.checkpoint_path
            model = utils.build_model(options['model'])
            try:
                os.makedirs(self.args.save_path, exist_ok=True)
            except:
                print('Invalid file path!')

            if self.input_path:
                self.mode = self.define_mode(self.input_path)
                if not self.mode:
                    self.textEdit.setText(f'This file extension is not supported')
                    return
                else:
                    self.textEdit.append('Start Conversion.. Please Wait..')
                    self.textEdit.repaint()
                    if self.mode=='image':
                        self.process_image(self.input_path, model)
                    else:
                        self.process_video(self.input_path, model)
                    self.textEdit.append(f'Conversion done!\n')
                    self.textEdit.append(f'Saved path : {self.args.save_path}')

            elif self.inputfolder_path:
                for idx, input_file in enumerate(os.listdir(self.inputfolder_path)):
                    self.mode = self.define_mode(input_file)
                    if not self.mode:
                        self.textEdit.setText(f'{idx}. This file extension is not supported')
                        continue
                    else:
                        self.textEdit.append(f'{idx}. Start Conversion.. Please Wait..')
                        self.textEdit.repaint()
                        if self.mode=='image':
                            self.process_image(self.inputfolder_path+'/'+input_file, model)
                        else:
                            self.process_video(self.inputfolder_path+'/'+input_file, model)
                        self.textEdit.append(f'Conversion done!\n')
                    time.sleep(3)
                self.textEdit.append(f'Saved path : {self.args.save_path}')


        except Exception as e:
            self.textEdit.append(f'Error Occured! : {e}')

        self.input_path = None
        self.inputfolder_path = None
        self.label.setText('')
        self.label_2.setText('')
        self.label_5.setText('')
        self.event_loop.exit()

    def display_image(self, path):
        self.image_file = QPixmap()
        self.image_file.load(path)
        # self.image_file = self.image_file.scaledToWidth(600)
        self.image_file = self.image_file.scaled(640, 650, Qt.KeepAspectRatio)
        self.label_3.setPixmap(self.image_file)
        self.label_3.show()
        self.event_loop.processEvents()
    def process_image(self, input_path, model):
        print("start....")
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
        # time.sleep(3)
        print("finish img conversion")

        self.display_image(os.path.join(self.args.save_path, 'compare', filename))

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

                im = qimage2ndarray.array2qimage(compare)
                im = QPixmap.fromImage(im)
                # self.image_file = self.image_file.scaledToWidth(600)
                im = im.scaled(640, 650, Qt.KeepAspectRatio)
                self.label_3.setPixmap(im)
                self.label_3.show()
                self.event_loop.processEvents()

            else:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break

        try:
            clip = ImageSequenceClip.ImageSequenceClip(new_frames, fps=fps)
            clip = clip.set_audio(audioclip)
            clip.write_videofile(os.path.join(args.save_path, filename), verbose=False, logger=None)

            clip = ImageSequenceClip.ImageSequenceClip(compare_frames, fps=fps)
            clip = clip.set_audio(audioclip)
            clip.write_videofile(os.path.join(self.args.save_path, 'compare', filename), verbose=False, logger=None)
        except Exception as e:
            pass
            # self.process_label.setText(f'{filename} video conversion has not completed due to : {e}')

        vidcap.release()


if __name__ == "__main__" :
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
    myWindow = WindowDiver(args)
    # myWindow.show()
    app.exec_()