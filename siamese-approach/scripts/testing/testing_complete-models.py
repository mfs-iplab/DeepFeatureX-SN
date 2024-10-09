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
    mydataset,
    dataset_for_robustness,
    make_balanced,
    balance_binary_test,
    get_trans
)
from dfx import test
from dfx import get_path


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str)
    parser.add_argument('--model_dict_path', type=str, default=None)
    parser.add_argument('--classification_type', type=str, default='binary', choices=['binary', 'multi-class'])
    parser.add_argument('--triplet', type=bool, default=False)
    parser.add_argument('--augmented', type=bool, default=False)
    parser.add_argument('--plot_cm', type=bool, default=False)
    parser.add_argument('--save_cm', type=bool, default=False)
    parser.add_argument('--average', type=str, default='binary', choices=['binary', 'micro', 'macro'])
    parser.add_argument('--test_raw', type=bool, default=False)
    parser.add_argument('--robustness_test', type=bool, default=False)
    parser.add_argument('-rt', '--robustness_types', nargs='+', choices=['jpegQF90','jpegQF80','jpegQF70','jpegQF60','jpegQF50', 'GaussNoise-3', 'GaussNoise-7', \
                                                        'GaussNoise-15', 'mir-B', 'rot-45', 'rot-135', 'scaling-50', 'scaling-200'])

    args = parser.parse_args()
    return args


def main(parser):

    datasets_path = get_path('dataset')
    guidance_path = get_path('guidance')
    models_dir = get_path('models')
    robustnessdset_path = get_path('data_robustness')

    complete_model = get_complete_triplet(parser.backbone, models_dir=models_dir, extracted_features=2208*3, dim_emb=128) if parser.triplet else get_complete_model(parser.backbone, models_dir=models_dir)
    
    if not parser.model_dict_path==None:
        complete_model.load_state_dict(torch.load(parser.model_dict_path))
    else:
        weight_path = models_dir+'/complete/'+parser.backbone+'.pt'
        if parser.triplet: weight_path = models_dir+'/triplet/complete/'+parser.backbone+'-tricomplete.pt'  
        if parser.augmented: weight_path = models_dir+'/complete/aug_'+parser.backbone+'.pt'
                    
        complete_model.load_state_dict(torch.load(weight_path))

    trans = get_trans(model_name=parser.backbone)

    if parser.test_raw:
        print('-    RAW')
        if parser.classification_type=='binary':
            testing = balance_binary_test(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_basemodel=False, for_testing=True, transforms=trans))
        else:
            testing = make_balanced(mydataset(dset_dir=datasets_path, guidance=guidance_path, for_basemodel=False, for_testing=True, transforms=trans))
        testload = DataLoader(testing, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
        loss = nn.CrossEntropyLoss()
        test(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=parser.plot_cm, save_cm=parser.save_cm, average=parser.average, convert_to_binary=True if parser.classification_type=='binary' else False, saving_dir='', model_name=parser.backbone)

    if parser.robustness_test:
        for testin in parser.robustness_types:
            print(f'-   {testin}')
            if parser.classification_type=='binary':
                testing = balance_binary_test(dataset_for_robustness(dset_dir=robustnessdset_path+f'/testing_dset-{testin}', transforms=trans))
            else:
                testing = make_balanced(dataset_for_robustness(dset_dir=robustnessdset_path+f'/testing_dset-{testin}', transforms=trans))

            testload = DataLoader(testing, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
            loss = nn.CrossEntropyLoss()

            test(model=complete_model, test_loader=testload, loss_fn=loss, plot_cm=parser.plot_cm, save_cm=parser.save_cm, average=parser.average, convert_to_binary=True if parser.classification_type=='binary' else False, saving_dir='', model_name=parser.backbone)

if __name__=='__main__':
    parser = get_parser()
    main(parser)