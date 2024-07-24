import torch
import torch.nn as nn
from torchvision.transforms import v2
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os, argparse

from dfx import (
    pair_dset,
    make_balanced,
    get_trans
)
from dfx import train_siamese
from dfx import ContrastiveLoss
from dfx import get_path
from dfx import backbone

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode_logs', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--main_class', type=str, choices=['dm_generated','gan_generated','real'])
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--scheduler_stepsize', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)

    args = parser.parse_args()
    return args


def main(parser):

    datasets_path = get_path('dataset')
    guidance_path = get_path('guidance')
    models_dir = get_path('models')

    conf = f'-b{parser.batch_size}-lr{parser.learning_rate}-wd{parser.weight_decay}-step{parser.scheduler_stepsize}-gamma{parser.scheduler_gamma}'

    batch_size = parser.batch_size

    trans = get_trans(model_name=parser.backbone,
                      transformations=[
                          v2.RandomChoice([v2.GaussianBlur(kernel_size=1), v2.GaussianBlur(kernel_size=3), v2.GaussianBlur(kernel_size=5), v2.GaussianBlur(kernel_size=9)]),
                          v2.RandomChoice([v2.RandomRotation(degrees=(0,360)), v2.RandomRotation(degrees=0)]),
                          v2.RandomChoice([v2.RandomResizedCrop(size=256), v2.RandomResizedCrop(size=512), v2.RandomResizedCrop(size=1024)]),
                          v2.RandomChoice([v2.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.5)]),
                          v2.RandomChoice([v2.JPEG(quality=100), v2.JPEG(quality=80), v2.JPEG(quality=60), v2.JPEG(quality=40), v2.JPEG(quality=20)])       
                        ]
                    )

    dset = make_balanced(pair_dset(dset_dir=datasets_path, main_class=parser.main_class, guidance=guidance_path, for_basemodel=True, for_testing=False, transforms=trans), binary=True)

    train, valid = random_split(dset, lengths=[.8,.2])

    trainload = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    validload = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    loss=ContrastiveLoss(m=2)

    backbone_name = parser.backbone

    def get_saving_dir(main_class):
        if main_class=='gan_generated': sv_dir='bm-gan'
        if main_class=='dm_generated': sv_dir='bm-dm'
        if main_class=='real': sv_dir='bm-real'
        return sv_dir

    saving_dir = os.path.join(models_dir, get_saving_dir(parser.main_class))+'/'
    models_sd_dir = os.path.join(models_dir, get_saving_dir(parser.main_class), backbone_name)+f'-{get_saving_dir(parser.main_class).split('-')[-1]}.pt'

    print(f'\n-  {backbone_name}\n')
    base_model = backbone(backbone_name, pretrained=True, finetuning=False, as_feature_extractor=True)
    base_model.load_state_dict(torch.load(models_sd_dir))

    optimizer = Adam(base_model.parameters(),
                    lr=parser.learning_rate, 
                    weight_decay=parser.weight_decay, 
                    betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=parser.scheduler_stepsize, gamma=parser.scheduler_gamma) if parser.scheduler else None

    train_siamese(model=base_model,
            loaders={'train': trainload, 'valid': validload},
            epochs=parser.epochs,
            optimizer=optimizer,
            loss_fn=loss,
            scheduler=scheduler,
            mode_logs=parser.mode_logs,
            model_name=backbone_name,
            save_best_model=parser.save_model,
            saving_path=saving_dir+'aug_'+backbone_name+conf+'.pt')

if __name__=='__main__':
    parser = get_parser()
    main(parser)