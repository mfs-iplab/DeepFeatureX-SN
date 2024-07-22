import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys, os, argparse

from dfx import get_encoder
from dfx import (
    triplet_dset,
    make_balanced,
    get_trans
)
from dfx import train_triplet_encoder
from dfx import get_path

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--dim_embeddings', type=int, default=512)
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
    
    dset = make_balanced(triplet_dset(dset_dir=datasets_path, guidance=guidance_path, for_basemodel=False, for_testing=False, transforms=trans))
    train_dset, valid_dset = random_split(dset, lengths=[.8,.2])

    trainload = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    validload = DataLoader(valid_dset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    loss=nn.TripletMarginLoss()

    backbone_name = parser.backbone

    print(f'\n-  {backbone_name}\n')
    triplet_encoder = get_encoder(backbone_name=backbone_name, models_dir=models_dir, extracted_features=2208*3, dim_emb=parser.dim_embeddings)

    optimizer = Adam(triplet_encoder.parameters(),
            lr=parser.learning_rate, 
            weight_decay=parser.weight_decay, 
            betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=parser.scheduler_stepsize, gamma=parser.scheduler_gamma) if parser.scheduler else None

    train_triplet_encoder(model=triplet_encoder,
        loaders={'train': trainload, 'valid': validload},
        epochs=parser.epochs,
        optimizer=optimizer,
        loss_fn=loss,
        scheduler=scheduler,
        mode_logs=parser.mode_logs,
        model_name=backbone_name,
        save_best_model=True,
        saving_path=os.path.join(models_dir,'triplet/encoder',backbone_name+conf+'.pt'))

if __name__=='__main__':
    parser = get_parser()
    main(parser)