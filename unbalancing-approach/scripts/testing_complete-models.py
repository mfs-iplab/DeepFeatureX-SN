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

# %%
parser = argparse.ArgumentParser(
    prog='training complete model',
    description='train complete model'
)

parser.add_argument('-b', '--backbone', type=str)
parser.add_argument('-back_path', '--backbone_path', type=str, default=None)
parser.add_argument('-cls_type_', '--classification_type', type=str, default='binary', choices=['binary', 'multi-class'])
parser.add_argument('-plt_cm', '--plot_cm', type=str, default=True)
parser.add_argument('-sv_cm', '--save_cm', type=str, default=False)
parser.add_argument('-ave', '--average', type=str, default='binary', choices=['binary', 'micro', 'macro'])
parser.add_argument('-robustness', '--robustness_test', type=bool, default=False)

args = parser.parse_args()

# %% load the complete models
complete_model = get_complete_model(args.backbone, models_dir=models_dir)
if not args.backbone_path==None:
    complete_model.load_state_dict(torch.load(args.backbone_path))
else:
    saved_backbone_name = call_saved_model(backbone_name=args.backbone)
    complete_model.load_state_dict(torch.load(models_dir+'/complete_models/'+saved_backbone_name+'.pt'))

trans = get_trans(model_name=args.backbone)

if args.classification_type=='binary':
    test = balance_binary_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans))
else:
    test = balance_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_overfitting=False, for_testing=True, transforms=trans))
testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
loss = nn.CrossEntropyLoss()

print('-    RAW')
testing(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=args.plot_cm, save_cm=args.save_cm, average=args.average, convert_to_binary=True if args.classification_type=='binary' else False, saving_dir=pictures_dir+'/cunfusion_matrices/RAW', model_name=args.backbone)

if args.robustness_test:
    for testin in ['jpegQF90','jpegQF80','jpegQF70','jpegQF60','jpegQF50']:
        print(f'-   {testin}')
        if args.classification_type=='binary':
            test = balance_binary_test(dataset_for_robustness(dset_dir=robustnessdset_path+f'/testing_dset-{testin}', transforms=trans))
        else:
            test = balance_test(dataset_for_robustness(dset_dir=robustnessdset_path+f'/testing_dset-{testin}', transforms=trans))

        testload = DataLoader(test, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        loss = nn.CrossEntropyLoss()

        testing(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=args.plot_cm, save_cm=args.save_cm, average=args.average, convert_to_binary=True if args.classification_type=='binary' else False, saving_dir=pictures_dir+'', model_name=args.backbone)