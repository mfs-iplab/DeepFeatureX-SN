import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dfx import (
    get_path,
    get_trans,
    make_balanced,
    mydataset,
    backbone,
)

def get_parser():
    """
    Set up and return the argument parser for the script.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=True, help='Name of the backbone model')
    parser.add_argument('--dimensions', type=int, choices=[2, 3], default=2, help='Number of dimensions for t-SNE (2 or 3)')
    parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity parameter for t-SNE')
    parser.add_argument('--n_iter', type=int, default=1000, help='Number of iterations for t-SNE')
    parser.add_argument('--model_dict_path', type=str, default=None, help='Custom path for model dictionary')
    parser.add_argument('--augmented', type=bool, default=False, help='Use augmented data')
    return parser.parse_args()

def perform_tsne(data, n_components, perplexity, n_iter):
    """
    Perform t-SNE dimensionality reduction.

    Args:
        data (np.array): Input data
        n_components (int): Number of dimensions for t-SNE output
        perplexity (float): Perplexity parameter for t-SNE
        n_iter (int): Number of iterations for t-SNE

    Returns:
        np.array: t-SNE transformed data
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    return tsne.fit_transform(data)

def plot_tsne(tsne_result, labels, backbone_name, folder, n_components):
    """
    Plot t-SNE results.

    Args:
        tsne_result (np.array): t-SNE transformed data
        labels (np.array): True labels for the data
        backbone_name (str): Name of the backbone model
        folder (str): Folder name (e.g., 'bm-dm', 'bm-gan', 'bm-real')
        n_components (int): Number of dimensions in t-SNE result
    """
    plt.figure(figsize=(10, 8))
    
    # Define class names and colors
    class_names = ['DM', 'GAN', 'REAL']
    colors = ['purple', 'green', 'yellow']
    
    if n_components == 2:
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], c=colors[i], label=class_name, s=10)
        plt.xlabel('t-SNE component 1', fontsize=8)
        plt.ylabel('t-SNE component 2', fontsize=8)
    else:  # 3D plot
        ax = plt.figure().add_subplot(111, projection='3d')
        for i, class_name in enumerate(class_names):
            mask = labels == i
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], tsne_result[mask, 2], c=colors[i], label=class_name, s=10)
        ax.set_xlabel('t-SNE component 1', fontsize=8)
        ax.set_ylabel('t-SNE component 2', fontsize=8)
        ax.set_zlabel('t-SNE component 3', fontsize=8)

    base_model_type = folder.split('-')[-1].upper()
    plt.title(f'{base_model_type}-BASE MODEL', fontsize=10)
    plt.legend(fontsize=8)
    
    plt.tick_params(axis='both', which='major', labelsize=8)
    
    save_path = f'tsne_plot_{backbone_name}_{folder}_{n_components}D.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def main(args):
    backbone_name = args.backbone
    datasets_path = get_path('dataset')
    guidance_path = get_path('guidance')
    models_dir = get_path('models')

    trans = get_trans(model_name=backbone_name)
    testing_dset = make_balanced(mydataset(datasets_path, guidance_path, for_basemodel=False, for_testing=True, transforms=trans))
    testloader = DataLoader(testing_dset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    folders = ['bm-dm', 'bm-gan', 'bm-real']

    for folder in tqdm(folders, desc="Processing folders"):
        model = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
        model.eval()
        if args.augmented: backbone_name = 'aug_' + backbone_name
        state_dict_path = os.path.join(models_dir, folder, backbone_name+folder[2:]+'.pt')
        if not args.model_dict_path==None: state_dict_path = args.model_dict_path
        model.load_state_dict(torch.load(state_dict_path))
        data, labels = [], []
        
        for batch in tqdm(testloader, desc=f"Processing {folder} data", leave=False):
            img, label, _ = batch
            with torch.no_grad():
                code = model(img)
            code = code.to('cpu').numpy()
            labels.append(label.numpy())
            data.append(code)
        
        data, labels = np.array(data), np.array(labels).squeeze()
        
        print(f"Performing t-SNE for {folder}...")
        tsne_result = perform_tsne(data.reshape(data.shape[0], -1), args.dimensions, args.perplexity, args.n_iter)
        
        plot_tsne(tsne_result, labels, backbone_name, folder, args.dimensions)

if __name__ == '__main__':
    args = get_parser()
    main(args)