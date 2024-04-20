# %% import libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys, os
from dataset_classes import *
from training_procedure import *
from import_classifiers import *
from settings import *
from architecture import *
from torch.utils.data import DataLoader

# %% dataset preparation
batch_size = 32

trans = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_vits = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dset = mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=False, transforms=trans_vits)
train, valid = make_train_valid(dset=dset, validation_ratio=0.2)

perc_dm, perc_gan, perc_real = check_len(dset, binary=False, return_perc=True)

trainload = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
validload = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

test = balance_binary_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans))
test_vits = balance_binary_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans_vits))
testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
testvitload = DataLoader(test_vits, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

loss = nn.CrossEntropyLoss(weight=torch.tensor([1-perc_dm,1-perc_gan,1-perc_real]).to(dev))

# %% 
models = ['vit_b_16', 'vit_b_32']

for model_name in models:
    model = backbone(model_name, pretrained=True, finetuning=True, num_classes=3)
    optimizer = Adam(model.parameters(), 
                    lr=1e-4, 
                    weight_decay=1e-2, 
                    betas=(0.9, 0.999))
    scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.9)

    training(model=model,
            loaders={'train': trainload, 'valid': validload},
            epochs=50,
            optimizer=optimizer,
            loss_fn=loss,
            scheduler=scheduler,
            logs=True,
            model_name=model_name+'-baseline',
            save_best_model=True,
            saving_path=models_dir+'/base/'+model_name+'.pt')
    testing(model=model, test_loader=testload, loss_fn=nn.CrossEntropyLoss(), plot_cm=True, average='micro')