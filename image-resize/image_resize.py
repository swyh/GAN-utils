import os
import cv2
import numpy as np
import argparse
from scipy import misc

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')

parser.add_argument('--input_path', type=str, default='./input', help='Set the input images path')
parser.add_argument('--output_path', type=str, default='./output', help='Set the output images path')
parser.add_argument('--width', type=int, default=64, help='Set the image_size')
parser.add_argument('--height', type=int, default=64, help='Set the image_size')

global args
args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
width = args.width
height = args.height


def read_and_write(input_path, output_path):
    images = []
    names = os.listdir(input_path)

    for file in names:
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG"):
            file_path = os.path.join(input_path, file)

            stream = open(file_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (width, height))

            images.append(image)

    if images:
        images = np.stack(images)
    else:
         return


    for i in range(0,len(images)):
        print("save image :", names[i])
        misc.imsave(os.path.join(output_path, names[i]), images[i])



def save_directory(input_path, output_path):
    names = os.listdir(input_path)
    for name in names:
        if os.path.isdir(os.path.join(input_path, name)):
            os.mkdir(os.path.join(output_path, name))
            save_directory(os.path.join(input_path, name),os.path.join(output_path, name))

    if os.path.isdir(os.path.join(input_path)):
        read_and_write(os.path.join(input_path), os.path.join(output_path))



if not os.path.isdir(output_path):
    os.mkdir(output_path)

save_directory(input_path, output_path)
