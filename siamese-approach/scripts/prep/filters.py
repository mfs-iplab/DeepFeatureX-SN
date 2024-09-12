import os
import cv2
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from dfx import get_path

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--filters', type=str, action='append', default=['-jpegQF90','-jpegQF80','-jpegQF70','-jpegQF60','-jpegQF50', \
                                                                        '-GaussNoise-3', '-GaussNoise-7', '-GaussNoise-15', \
                                                                        '-scaling-50', '-scaling-200', \
                                                                        '-mir-B', '-rot-45', '-rot-135'])
    args = parser.parse_args()

    return args


def Rotation(image_path, path_dir, image, fold, rotations: list | None = [45, 135, 225, 315]):
    for rotation in rotations:
        img = cv2.imread(image_path)
        h, w, _= img.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        rotated_image = cv2.warpAffine(img, M, (w, h))
        new_img_path = path_dir+"/dset-rot-"+str(rotation)+"/"+fold+"/"+image[:-4]+".png"
        if not os.path.exists(new_img_path):
            cv2.imwrite(new_img_path, rotated_image)


def Mirror(image_path, path_dir, image, fold):
    img = cv2.imread(image_path)
    flipBoth = cv2.flip(img, -1)
    new_img_path = path_dir+"/dset-mir-B/"+fold+"/"+image[:-4]+".png"
    if not os.path.exists(new_img_path):
        cv2.imwrite(new_img_path, flipBoth)
        

def Scaling(image_path, path_dir, image, fold, factors: list | None = [50, 200]):
    for scale_percent in factors:
        img = cv2.imread(image_path)
        width, height = int(img.shape[1] * scale_percent / 100), int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        new_img_path = path_dir+"/dset-scaling-"+str(scale_percent)+"/"+fold+"/"+image[:-4]+".png"
        if not os.path.exists(new_img_path):
            cv2.imwrite(new_img_path, resized_image)


def GaussNoise(image_path, path_dir, image, fold, kernel: list | None =[3,7,9,15]):
    for k in kernel:
        img = cv2.imread(image_path)
        dst = cv2.GaussianBlur(img,(k,k),cv2.BORDER_DEFAULT)
        new_img_path = path_dir+"/dset-GaussNoise-"+str(k)+"/"+fold+"/"+image[:-4]+".png"
        if not os.path.exists(new_img_path):
            cv2.imwrite(new_img_path, dst)


def JPEGCompr(image_path, path_dir, image, fold, qfac: list | None = [1,10,20,30,40,50,60,70,80,90]):
    for q in qfac:
        im = Image.open(image_path)
        new_img_path = path_dir+"/dset-jpegQF"+str(q)+"/"+fold+"/"+image[:-4]+'.jpg'
        if not os.path.exists(new_img_path):
            im.save(new_img_path,format='jpeg', subsampling=0, quality=q)


def main(parser):
    path_dset = get_path('dataset')
    path_augdset = get_path('augmented_dataset')
    guidance_path = get_path('guidance')
    path_test = get_path('data_robustness')

    filters = parser.filters

    path_dir = path_test if not parser.train else path_augdset

    for fil in filters:
        if not os.path.exists(os.path.join(path_dir, f'dset{fil}')):
            os.makedirs(os.path.join(path_dir, f'dset{fil}'))
        for architecture in os.listdir(path_dset):
            architecture_path = os.path.join(path_dset, architecture)
            for fold in os.listdir(architecture_path):
                if not os.path.exists(os.path.join(path_dir, f'dset{fil}', architecture, fold)):
                    os.makedirs(os.path.join(path_dir, f'dset{fil}', architecture, fold))

    guidance_csv = pd.read_csv(guidance_path)
    n = 2 if not parser.train else 1
    df = guidance_csv[guidance_csv['label']==n]
    progressive_bar = tqdm(df.iterrows(), total=len(df))
    progressive_bar.desc = 'processing images'
    for _, row in progressive_bar:
        img_path = path_dset+row['image_path']
        img = img_path.split('/')[-1]
        fold = os.path.join(img_path.split('/')[-3], img_path.split('/')[-2])
        JPEGCompr(img_path, path_dir, img, fold, qfac=[90,80,70,60,50])
        GaussNoise(img_path, path_dir, img, fold, kernel=[3,7,15])
        Scaling(img_path, path_dir, img, fold, factors=[50,200])
        Mirror(img_path, path_dir, img, fold)
        Rotation(img_path, path_dir, img, fold, rotations=[45,135])

if __name__=='__main__':
    parser = get_parser()
    main(parser)