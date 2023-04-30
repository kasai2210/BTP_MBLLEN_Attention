import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K
from keras.models import model_from_json
from keras.models import load_model
from glob import glob
import numpy as np
import scipy
import keras
import os
import Network
import utls
import time
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default='../input', help='test image folder')
parser.add_argument("--result", "-r", type=str, default='../result', help='result folder')
parser.add_argument("--model", "-m", type=str, default='10_dark_base', help='model name')
parser.add_argument("--com", "-c", type=int, default=0, help='output with/without origional image and mid result')
parser.add_argument("--highpercent", "-hp", type=int, default=95, help='should be in [85,100], linear amplification')
parser.add_argument("--lowpercent", "-lp", type=int, default=5, help='should be in [0,15], rescale the range [p%,1] to [0, 1]')
parser.add_argument("--gamma", "-g", type=int, default=8, help='should be in [6,10], increase the saturability')
parser.add_argument("--maxrange", "-mr", type=int, default=8, help='linear amplification range')
arg = parser.parse_args()

model_name = arg.model
mbllen = Network.build_mbllen((None, None, 3))
# mbllen = Network.build_vgg()
mbllen.load_weights('../models/'+model_name+'.h5')
opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
mbllen.compile(loss='mse', optimizer=opt)
# model = load_model("model.h5")
flag = arg.com
lowpercent = arg.lowpercent
highpercent = arg.highpercent
maxrange = arg.maxrange/10.
hsvgamma = arg.gamma/10.

result_folder = arg.result
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

top=tk.Tk()
top.geometry('1200x600')
top.title('Low Light Image Enhancement using CNNs')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',17,'bold'))
sign_image = Label(top)
sign_image1 = Label(top)

def convert(file_path):

    # The original image
    uploaded=Image.open(file_path)
    uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
    im=ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image=im
    sign_image.place(relx=0.1, rely=0.3)


    global label_packed
    img_A = utls.imread_color(file_path)
    img_A = cv2.resize(img_A, (256, 256))
    img_A = img_A[np.newaxis, :]

    out_pred = mbllen.predict(img_A, batch_size=1)
    # print("Hello world \n")
    fake_B = out_pred[0, :, :, :3]
    fake_B_o = fake_B

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
    percent_max = sum(sum(gray_fake_B >= maxrange))/sum(sum(gray_fake_B <= 1.0))
    # print(percent_max)
    max_value = np.percentile(gray_fake_B[:], highpercent)
    if percent_max < (100-highpercent)/100.:
        scale = maxrange / max_value
        fake_B = fake_B * scale
        fake_B = np.minimum(fake_B, 1.0)

    gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
    sub_value = np.percentile(gray_fake_B[:], lowpercent)
    fake_B = (fake_B - sub_value)*(1./(1-sub_value))

    imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    fake_B = np.minimum(fake_B, 1.0)

    if flag:
        outputs = np.concatenate([img_A[0,:,:,:], fake_B_o, fake_B], axis=1)
    else:
        outputs = fake_B

    filename = os.path.basename(file_path)
    img_name = result_folder+'/' + filename
    outputs = np.minimum(outputs, 1.0)
    outputs = np.maximum(outputs, 0.0)
    utls.imwrite(img_name, outputs)

    # The converted image
    converted=Image.open(img_name)
    converted.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
    im1=ImageTk.PhotoImage(converted)
    sign_image1.configure(image=im1)
    sign_image1.image=im1
    sign_image1.place(relx=0.6, rely=0.3)

    
def show_convert_button(file_path):
    convert_b=Button(top,text="Convert Image", command=lambda: convert(file_path),padx=10,pady=5)
    convert_b.configure(background='#364156', foreground='white', font=('arial',14,'bold'))
    convert_b.place(relx=0.421, rely=0.14)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        sign_image.place(relx=0.12, rely=0.3)
        label.configure(text='')
        show_convert_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image, padx=10,pady=5)
upload.configure(background='#364156', foreground='white', font=('arial',14,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Low Light Image Enhancement using CNNs",pady=20, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()