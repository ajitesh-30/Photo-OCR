import argparse 
import pandas as pd
import numpy as np
import itertools
from PIL import Image
import pytesseract
import argparse
import cv2
import os
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
#import Digit_recognition_CNN.py
#import Multidigit_web_cam.py

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")

args = vars(ap.parse_args())
def get_text_file():
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	if args["preprocess"] == "thresh":
		gray = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
	elif args["preprocess"] == "blur":
		gray = cv2.medianBlur(gray, 3)
	
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)
	text = pytesseract.image_to_string(Image.open(filename)) 
	
	os.remove(filename)
	f= open("result.txt","w+")
	f.write(text)
	print('Text File Created')

get_text_file()






