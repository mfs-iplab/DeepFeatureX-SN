# %%
import torch
import torch.nn as nn
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
# %%
parser = argparse.ArgumentParser()

parser.add_argument('-b', '--backbone', type=str)
parser.add_argument('-logs', '--mode_logs', type=str, default='online')
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
def main():
        batch_size = args.batch_size

        trans = get_trans(model_name=args.backbone)
        
        dset = mydataset(dset_dir=datasets_path, guidance=guidance_path, for_basemodel=False, for_testing=False, transforms=trans)
        trainset, validset = make_train_valid(dset=dset, validation_ratio=0.2)

        perc_dm, perc_gan, perc_real = check_len(dset, binary=False, return_perc=True)

        trainload = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        validload = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        loss = nn.CrossEntropyLoss(weight=torch.tensor([1-perc_dm,1-perc_gan,1-perc_real]).to(dev))

        backbone_name = args.backbone

        print(f'\n-  {backbone_name}\n')
        model_complete = get_compete_triplet(backbone_name=backbone_name, models_dir=models_dir, extracted_features=2208*3, dim_emb=128)

        optimizer = Adam(model_complete.parameters(), 
                        lr=args.learning_rate, 
                        weight_decay=args.weight_decay, 
                        betas=(0.9, 0.999))
        scheduler = StepLR(optimizer=optimizer, step_size=args.scheduler_stepsize, gamma=args.scheduler_gamma) if args.scheduler else None

        train(model=model_complete,
                loaders={'train': trainload, 'valid': validload},
                epochs=args.epochs,
                optimizer=optimizer,
                loss_fn=loss,
                scheduler=scheduler,
                mode_logs=args.mode_logs,
                model_name=backbone_name,
                save_best_model=True,
                saving_path=os.path.join(models_dir,'triplet/complete',backbone_name+conf+'.pt'))

if __name__=='__main__':
        main()