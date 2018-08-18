import os
import cv2
import numpy as np
import argparse
from scipy import misc

parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')

parser.add_argument('--input_path', type=str, default='./input', help='Set the input images path')
parser.add_argument('--output_path', type=str, default='./output', help='Set the output images path')
parser.add_argument('--width', type=int, default=256, help='Set the image_size')
parser.add_argument('--height', type=int, default=256, help='Set the image_size')
parser.add_argument('--dir_A', type=str, default='./dir_A', help='Set the derectory_A')
parser.add_argument('--dir_B', type=str, default='./dir_B', help='Set the derectory_B')

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
width = args.width
height = args.height
image_cnt = 0


def read_image(input_path):
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
        return images
    else:
        return None


def read_directory(input_path): # 디렉토리 안의 파일 모두 저장 후 반환
    names = os.listdir(input_path)
    data = []

    for name in names:
        print("      - dir :", name)
        if os.path.isdir(os.path.join(input_path, name)):
            data.append(read_image(os.path.join(input_path, name)))

    return data


def save_combine(A,B):
    print("combine A & B")

    for i in range(0,min(len(A), len(B))):
        for j in range(0, min(len(A[i]), len(B[i]))):
            global image_cnt

            image = np.hstack([A[i][j], B[i][j]])
            print(image_cnt, "save image")
            misc.imsave(os.path.join(args.output_path, str(image_cnt) + ".png"), image)
            image_cnt = image_cnt + 1


def search_directory(input_path, name):
    print("current dir :", name)
    names = os.listdir(input_path)

    datasetA = []
    datasetB = []

    for name in names:
        path = os.path.join(input_path, name)
        if os.path.isdir(path):
            if name == args.dir_A: # dataset A
                datasetA = read_directory(path)
            elif name == args.dir_B:  # dataset B
                datasetB = read_directory(path)
            else:
                search_directory(os.path.join(input_path, name), name)

    if datasetA is not None:
        save_combine(datasetA, datasetB)





if not os.path.isdir(output_path):
    os.mkdir(output_path)

search_directory(input_path, input_path)
