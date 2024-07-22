import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys, os, argparse

from dfx import get_complete_triplet
from dfx import (
    mydataset,
    check_len,
    make_train_valid,
    get_trans
)
from dfx import train
from dfx import get_path

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--mode_logs', type=str, default='online')
    parser.add_argument('--batch_size', type=int, default=32)
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

    trans = get_trans(model_name=parser.backbone)
    
    dset = mydataset(dset_dir=datasets_path, guidance=guidance_path, for_basemodel=False, for_testing=False, transforms=trans)
    trainset, validset = make_train_valid(dset=dset, validation_ratio=0.2)

    perc_dm, perc_gan, perc_real = check_len(dset, binary=False, return_perc=True)

    trainload = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    validload = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    loss = nn.CrossEntropyLoss(weight=torch.tensor([1-perc_dm,1-perc_gan,1-perc_real]).to(dev))

    backbone_name = parser.backbone

    print(f'\n-  {backbone_name}\n')
    model_complete = get_complete_triplet(backbone_name=backbone_name, models_dir=models_dir, extracted_features=2208*3, dim_emb=128)

    optimizer = Adam(model_complete.parameters(), 
            lr=parser.learning_rate, 
            weight_decay=parser.weight_decay, 
            betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=parser.scheduler_stepsize, gamma=parser.scheduler_gamma) if parser.scheduler else None

    train(model=model_complete,
        loaders={'train': trainload, 'valid': validload},
        epochs=parser.epochs,
        optimizer=optimizer,
        loss_fn=loss,
        scheduler=scheduler,
        mode_logs=parser.mode_logs,
        model_name=backbone_name,
        save_best_model=True,
        saving_path=os.path.join(models_dir,'triplet/complete',backbone_name+conf+'.pt'))

if __name__=='__main__':
    parser = get_parser()
    main(parser)