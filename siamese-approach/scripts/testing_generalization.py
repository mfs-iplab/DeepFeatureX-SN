# %% import libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings, argparse
warnings.filterwarnings('ignore')
from dataset_classes import *
from training_procedure import *
from import_classifiers import *
from settings import *
from architecture import *
from torch.utils.data import DataLoader

# %% load the best complete models
# %%
parser = argparse.ArgumentParser()

parser.add_argument('-b', '--backbone', type=str)
parser.add_argument('-model_type', '--model_type', type=str, default='complete', choices=['complete', 'backbone'])
parser.add_argument('-back_path', '--backbone_path', type=str, default=None)
parser.add_argument('-plt_cm', '--plot_cm', type=str, default=True)
parser.add_argument('-sv_cm', '--save_cm', type=str, default=False)
parser.add_argument('-ave', '--average', type=str, default='binary', choices=['binary', 'micro', 'macro'])

args = parser.parse_args()

# %% testing phase on RAW images
complete_model = get_complete_model(args.backbone, models_dir=models_dir) if args.model_type=='complete' else backbone(args.backbone, finetuning=True, num_classes=3)
if not args.backbone_path==None:
    complete_model.load_state_dict(torch.load(args.backbone_path))
else:
    saved_backbone_name = call_saved_model(backbone_name=args.backbone)
    complete_model.load_state_dict(torch.load(models_dir+'/complete_models/'+saved_backbone_name+'.pt'))

trans = get_trans(model_name=args.backbone)
loss = nn.CrossEntropyLoss()

for folder in ['inner_gan', 'outer_gan', 'io_gan', 'inner_dm', 'outer_dm', 'io_dm', 'inner_all', 'outer_all', 'all']:
    print(f'-   {folder}')
    test = make_binary(dataset_for_generaization(dset_dir=generalization_path+f'/{folder}', transforms=trans))
    testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    loss = nn.CrossEntropyLoss()
    testing(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=args.plot_cm, save_cm=args.save_cm, average=args.average, convert_to_binary=True, saving_dir=pictures_dir+'', model_name=args.backbone)

# %%
