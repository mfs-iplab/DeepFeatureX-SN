#%% import libraries
import torch
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys, os, argparse
from dataset_classes import *
from import_classifiers import *
from training_procedure import *
from architecture import *
from settings import *

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--backbone', type=str)
parser.add_argument('-logs', '--mode_logs', type=str, default='online', choices=['online', 'offline', 'disabled'])
parser.add_argument('-main', '--main_class', type=str, choices=['dm_generated','gan_generated','real'])
parser.add_argument('-save', '--save_model', type=bool, default=True)
parser.add_argument('-batch', '--batch_size', type=int, default=32)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-sch', '--scheduler', type=bool, default=False)
parser.add_argument('-sch_step', '--scheduler_stepsize', type=int, default=10)
parser.add_argument('-sch_g', '--scheduler_gamma', type=float, default=0.1)

args = parser.parse_args()
conf = f'-b{args.batch_size}-lr{args.learning_rate}-wd{args.weight_decay}-step{args.scheduler_stepsize}-gamma{args.scheduler_gamma}'

# %% dataset preparation
batch_size = args.batch_size

trans = get_trans(model_name=args.backbone)

dset = make_balanced(pair_dset(dset_dir=datasets_path, main_class=args.main_class, guidance=guidance_path, for_basemodel=True, for_testing=False, transforms=trans), binary=True)

train, valid = random_split(dset, lengths=[.8,.2])

trainload = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
validload = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

loss=ContrastiveLoss(m=2)

# %% training procedure
backbone_name = args.backbone

print(f'\n-  {backbone_name}\n')
base_model = backbone(backbone_name, pretrained=True, finetuning=False, as_feature_extractor=True)

optimizer = Adam(base_model.parameters(),
                lr=args.learning_rate, 
                weight_decay=args.weight_decay, 
                betas=(0.9, 0.999))
scheduler = StepLR(optimizer=optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma) if args.scheduler else None

def get_saving_dir(main_class):
    if main_class=='gan_generated': sv_dir='bm-gan'
    if main_class=='dm_generated': sv_dir='bm-dm'
    if main_class=='real': sv_dir='bm-real'
    return sv_dir

saving_dir = os.path.join(models_dir, get_saving_dir(args.main_class))+'/'

train_siamese(model=base_model,
        loaders={'train': trainload, 'valid': validload},
        epochs=args.epochs,
        optimizer=optimizer,
        loss_fn=loss,
        scheduler=scheduler,
        mode_logs=args.mode_logs,
        model_name=backbone_name,
        save_best_model=args.save_model,
        saving_path=saving_dir+backbone_name+conf+'.pt')