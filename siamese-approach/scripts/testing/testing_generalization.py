import torch
import torch.nn as nn
import torchvision.transforms as T
import warnings, argparse
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader

from dfx import (
    get_complete_model,
    get_complete_triplet
)
from dfx import (
    dataset_for_generaization,
    make_binary,
    get_trans
)
from dfx import test
from dfx import backbone
from dfx import get_path

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--model_type', type=str, default='complete', choices=['complete', 'backbone'])
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--backbone_path', type=str, default=None)
    parser.add_argument('--plot_cm', action='store_true')
    parser.add_argument('--save_cm', action='store_true')
    parser.add_argument('--average', type=str, default='binary', choices=['binary', 'micro', 'macro'])

    args = parser.parse_args()
    return args


def main(parser):

    models_dir = get_path('models')
    generalization_path = get_path('data_generalization')
    
    if parser.triplet:
        complete_model = get_complete_triplet(parser.backbone, models_dir=models_dir, extracted_features=2208*3, dim_emb=128)
    else:
        complete_model = get_complete_model(parser.backbone, models_dir=models_dir) if parser.model_type=='complete' else backbone(parser.backbone, finetuning=True, num_classes=3)

    if not parser.backbone_path==None:
        complete_model.load_state_dict(torch.load(parser.backbone_path))
    else:
        weight_path = models_dir+'/triplet/complete/'+parser.backbone+'-tricomplete.pt' if parser.triplet else models_dir+'/complete/'+parser.backbone+'.pt'
        complete_model.load_state_dict(torch.load(weight_path))

    trans = get_trans(model_name=parser.backbone)
    loss = nn.CrossEntropyLoss()

    for folder in ['inner_gan', 'outer_gan', 'io_gan', 'inner_dm', 'outer_dm', 'io_dm', 'inner_all', 'outer_all', 'all']:
        print(f'-   {folder}')
        testing = make_binary(dataset_for_generaization(dset_dir=generalization_path+f'/{folder}', transforms=trans))
        testload = DataLoader(testing, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

        loss = nn.CrossEntropyLoss()
        test(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=parser.plot_cm, save_cm=parser.save_cm, average=parser.average, convert_to_binary=True, saving_dir='', model_name=parser.backbone)

if __name__=='__main__':
    parser = get_parser()
    main(parser)