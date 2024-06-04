import os
import torch
import torch.nn as nn
from import_classifiers import *


class completenn(nn.Module):
    def __init__(self, model1, model2, model3):
        super(completenn, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        code1 = self.model1(x)
        code2 = self.model2(x)
        code3 = self.model3(x)
        x = torch.cat((code1.unsqueeze(1), code2.unsqueeze(1), code3.unsqueeze(1)), 1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class encoder_triplet(nn.Module):
    def __init__(self, model1, model2, model3, extracted_features, dim_emb = 512):
        super(encoder_triplet, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.encoder = nn.Sequential(
            nn.Linear(extracted_features, dim_emb),
            nn.SELU()
        )
    def forward(self, x):
        code1 = self.model1(x)
        code2 = self.model2(x)
        code3 = self.model3(x)
        feature_vector = torch.cat([code1, code2, code3], dim=1)
        out = self.encoder(feature_vector)
        return out

class complete_triplet(nn.Module):
    def __init__(self, encoder, dim_emb = 512):
        super(complete_triplet, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(dim_emb, 3)
        )
    def forward(self, x):
        code = self.encoder(x)
        out = self.classifier(code)
        return out
        
def get_encoder(backbone_name: str, models_dir:str, extracted_features, dim_emb = 512):
    model_dm = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
    model_gan = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
    model_real = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
    model_dm.load_state_dict(torch.load(os.path.join(models_dir, 'bm-dm', backbone_name+'-dm.pt')))
    model_gan.load_state_dict(torch.load(os.path.join(models_dir, 'bm-gan', backbone_name+'-gan.pt')))
    model_real.load_state_dict(torch.load(os.path.join(models_dir, 'bm-real', backbone_name+'-real.pt')))
    model_dm.eval()
    model_gan.eval()
    model_real.eval()
    encoder = encoder_triplet(model_dm, model_gan, model_real, extracted_features=extracted_features, dim_emb=dim_emb)
    for backbone_model in [encoder.model1, encoder.model2, encoder.model3]:
        for param in backbone_model.parameters():
            param.requires_grad = False
    return encoder

def get_model_families():
    model_families = {'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        'inception': ['googlenet', 'inception_v3'],
                        'resnet1': ['resnet18', 'resnet34', 'resnet50'],
                        'resnet2': ['resnet101', 'resnet152'],
                        'resnext': ['resnext101'],
                        'efficient' : ['efficientnet_b0', 'efficientnet_b4'],
                        'efficient_w': ['efficientnet_widese_b0', 'efficientnet_widese_b4'],
                        'vit_b': ['vit_b_16', 'vit_b_32'],
                        'vit_l': ['vit_l_16', 'vit_l_32']}
    return model_families

def get_complete_model(backbone_name: str, models_dir:str):
    model_dm = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
    model_gan = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
    model_real = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
    model_dm.load_state_dict(torch.load(os.path.join(models_dir, 'bm-dm', backbone_name+'-dm.pt')))
    model_gan.load_state_dict(torch.load(os.path.join(models_dir, 'bm-gan', backbone_name+'-gan.pt')))
    model_real.load_state_dict(torch.load(os.path.join(models_dir, 'bm-real', backbone_name+'-real.pt')))
    model_dm.eval()
    model_gan.eval()
    model_real.eval()
    complete_model = completenn(model_dm, model_gan, model_real)
    for backbone_model in [complete_model.model1, complete_model.model2, complete_model.model3]:
        for param in backbone_model.parameters():
            param.requires_grad = False
    return complete_model

def get_compete_triplet(backbone_name: str, models_dir:str, extracted_features, dim_emb = 512):
    triencoder = get_encoder(backbone_name=backbone_name, models_dir=models_dir, extracted_features=extracted_features, dim_emb = dim_emb)
    triplet_model = complete_triplet(encoder=triencoder, dim_emb=dim_emb)
    for param in triplet_model.encoder.parameters():
            param.requires_grad = False
    return triplet_model