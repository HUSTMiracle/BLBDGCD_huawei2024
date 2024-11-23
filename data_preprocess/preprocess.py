import argparse
import os
import random

from PIL import Image

from util_preprocess import *


def noprocess(img):
    return img


def data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, function, func_name):
    print("Processing " + os.path.join(folder, img_path))
    img = Image.open(os.path.join(folder, img_path))
    img = function(img)
    img_List = square_img_segmentation(img, 512, 512, 0.1)

    processed_img_list = []
    for i in range(len(img_List)):
        txt_path = txt_folder + img_path[:-3] + "txt"
        update_label(img, img_List[i], txt_path)
        if len(img_List[i].error_list) != 0:
            processed_img_list.append(img_List[i])

    for i in range(len(processed_img_list)):
        processed_img_list[i].img.save(image_output_folder + img_path[:-4] + '_' + func_name + '_' + str(i) + '.bmp')
        with open(txt_output_folder + img_path[:-4] + '_' + func_name + '_' + str(i) + '.txt', 'w') as file:
            for row in processed_img_list[i].error_list:
                file.write(' '.join(str(value) for value in row) + '\n')


def everytricks(folder, txt_folder, image_output_folder, txt_output_folder):
    Img_list = os.listdir(folder)
    for img_path in Img_list:

        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, adjust_brightness_contrast,
                     'brighter')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, add_gaussian_noise,
                     'gaussian')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, todarkblue, 'darkblue')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, noprocess, '')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, toblue, 'blue')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, togreen, 'green')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, tored, 'red')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, toorange, 'orange')
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, topurple, 'purple')

        img_routine = os.path.join(folder, img_path)
        txt_path = os.path.join(txt_folder, img_path[:-3] + "txt")
        img_output_path = os.path.join(image_output_folder, img_path)
        txt_output_path = os.path.join(txt_output_folder, img_path[:-3] + "txt")
        img_narrow(img_routine, txt_path, img_output_path, txt_output_path)
        img_enlarge(img_routine, txt_path, img_output_path, txt_output_path)
        img_shrink(img_routine, txt_path, img_output_path, txt_output_path)

def randomtricks(folder, txt_folder, image_output_folder, txt_output_folder, random_ratio=0.2):
    Img_list = os.listdir(folder)
    for img_path in Img_list:
        data_process(folder, txt_folder, image_output_folder, txt_output_folder, img_path, noprocess, 'random')
    des_list = os.listdir(image_output_folder)
    for img_name in des_list:
        if 'random' not in img_name:
            continue
        ratio = random.random()
        if ratio > random_ratio:
            continue
        img = Image.open(os.path.join(image_output_folder, img_name))
        img_path = os.path.join(image_output_folder, img_name)
        r = random.randint(0, 9)
        print('Processing', img_path, 'random state', r)
        if r == 0:
            img = todarkblue(img)
            img.save(img_path)
        else:
            rr = random.randint(0, 2)
            gr = random.randint(0, 2)
            br = random.randint(0, 2)
            if rr == gr and rr == br and r < 7:
                continue
            img = tocolor(img, rr, gr, br)
            img = img.save(img_path)
        txt_path = os.path.join(txt_output_folder, img_name.replace("bmp","txt"))
        rotate_image(img_path, txt_path, 90*(r%3), img_path, txt_path)


def right_process(folder, txt_folder, image_output_folder, txt_output_folder):
    Img_list = os.listdir(folder)
    for img_path in Img_list:
        print("Processing " + os.path.join(folder, img_path))
        img = Image.open(os.path.join(folder, img_path))
        img_List = square_img_segmentation(img, 512, 512, 0.1)

        processed_img_list = []
        for i in range(len(img_List)):
            txt_path = txt_folder + img_path[:-3] + "txt"
            update_label(img, img_List[i], txt_path)
            if len(img_List[i].error_list) != 0:
                continue
            else:
                img_List[i].img.save(image_output_folder + img_path[:-4] + '_right_' + str(i) + '.bmp')
                with open(txt_output_folder + img_path[:-4] + '_right_' + str(i) + '.txt', 'w') as file:
                    for row in img_List[i].error_list:
                        file.write(' '.join(str(value) for value in row) + '\n')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./train/images/")
    parser.add_argument("--txt_folder", type=str, default="./train/labels/")
    parser.add_argument("--image_output_folder", type=str, default="./train/output/images/")
    parser.add_argument("--txt_output_folder", type=str, default="./train/output/labels/")
    parser.add_argument("--mode", type=str, default="every", help='every, random or right')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    opt = parse()
    if opt.mode == "every":
        everytricks(opt.folder, opt.txt_folder, opt.image_output_folder, opt.txt_output_folder)
    elif opt.mode == "random":
        randomtricks(opt.folder, opt.txt_folder, opt.image_output_folder, opt.txt_output_folder)
    elif opt.mode == "right":
        right_process(opt.folder, opt.txt_folder, opt.image_output_folder, opt.txt_output_folder)
    else:
        print("Wrong mode")
