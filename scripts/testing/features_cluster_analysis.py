import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
from umap import UMAP
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
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
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--metrics', nargs='+', choices=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'wcss'], default=['silhouette'])
    parser.add_argument('--model_dict_path', type=str, default=None)
    parser.add_argument('--augmented', type=bool, default=False)
    return parser.parse_args()

def calculate_silhouette_scores(data, labels, n_components_range):
    """
    Calculate silhouette scores for different numbers of UMAP components.
    
    The silhouette score measures how similar an object is to its own 
    cluster compared to other clusters. A higher silhouette score indicates 
    better-defined clusters.

    Args:
        data (np.array): Input data
        labels (np.array): True labels for the data
        n_components_range (range): Range of UMAP components to test

    Returns:
        list: Silhouette scores for each number of components
    """
    scores = []
    for n_components in tqdm(n_components_range, desc="Calculating silhouette scores", leave=False):
        umap = UMAP(n_components=n_components, metric='cosine', random_state=42)
        umap_res = umap.fit_transform(data.squeeze())
        score = silhouette_score(umap_res, labels)
        scores.append(score)
    return scores

def calculate_davies_bouldin_scores(data, labels, n_components_range):
    """
    Calculate Davies-Bouldin scores for different numbers of UMAP components.
    
    This index signifies the average 'similarity' between clusters, where similarity 
    is a measure that compares the distance between clusters with the size of the clusters 
    themselves. Lower values indicate better clustering.

    Args:
        data (np.array): Input data
        labels (np.array): True labels for the data
        n_components_range (range): Range of UMAP components to test

    Returns:
        list: Davies-Bouldin scores for each number of components
    """
    scores = []
    for n_components in tqdm(n_components_range, desc="Calculating Davies-Bouldin scores", leave=False):
        umap = UMAP(n_components=n_components, metric='cosine', random_state=42)
        umap_res = umap.fit_transform(data.squeeze())
        score = davies_bouldin_score(umap_res, labels)
        scores.append(score)
    return scores

def calculate_calinski_harabasz_scores(data, labels, n_components_range):
    """
    Calculate Calinski-Harabasz scores for different numbers of UMAP components.
    
    Also known as the Variance Ratio Criterion. The score is defined as the ratio between 
    the within-cluster dispersion and the between-cluster dispersion. Higher values indicate 
    better-defined clusters.

    Args:
        data (np.array): Input data
        labels (np.array): True labels for the data
        n_components_range (range): Range of UMAP components to test

    Returns:
        list: Calinski-Harabasz scores for each number of components
    """
    scores = []
    for n_components in tqdm(n_components_range, desc="Calculating Calinski-Harabasz scores", leave=False):
        umap = UMAP(n_components=n_components, metric='cosine', random_state=42)
        umap_res = umap.fit_transform(data.squeeze())
        score = calinski_harabasz_score(umap_res, labels)
        scores.append(score)
    return scores

def calculate_wcss(data, n_components_range, n_clusters=3):
    """
    Calculate Within-Cluster Sum of Squares (WCSS) for different numbers of UMAP components.
    
    This measures the compactness of the clustering and can be used to help find the optimal number of clusters.

    Args:
        data (np.array): Input data
        n_components_range (range): Range of UMAP components to test
        n_clusters (int, optional): Number of clusters for K-means. Defaults to 3.

    Returns:
        list: WCSS scores for each number of components
    """
    scores = []
    for n_components in tqdm(n_components_range, desc="Calculating WCSS", leave=False):
        umap = UMAP(n_components=n_components, metric='cosine', random_state=42)
        umap_res = umap.fit_transform(data.squeeze())
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(umap_res)
        scores.append(kmeans.inertia_)
    return scores

def calculate_metrics(data, labels, n_components_range, metrics):
    """
    Calculate specified clustering metrics for different numbers of UMAP components.

    Args:
        data (np.array): Input data
        labels (np.array): True labels for the data
        n_components_range (range): Range of UMAP components to test
        metrics (list): List of metrics to calculate

    Returns:
        dict: Dictionary of calculated metrics, where keys are metric names and values are lists of scores
    """
    results = {}
    if 'silhouette' in metrics:
        results['Silhouette Score'] = calculate_silhouette_scores(data, labels, n_components_range)
    if 'davies_bouldin' in metrics:
        results['Davies-Bouldin Index'] = calculate_davies_bouldin_scores(data, labels, n_components_range)
    if 'calinski_harabasz' in metrics:
        results['Calinski-Harabasz Index'] = calculate_calinski_harabasz_scores(data, labels, n_components_range)
    if 'wcss' in metrics:
        results['WCSS'] = calculate_wcss(data, n_components_range)
    return results

def main(args):
    backbone_name = args.backbone
    datasets_path = get_path('dataset')
    guidance_path = get_path('guidance')
    models_dir = get_path('models')

    trans = get_trans(model_name=backbone_name)
    testing_dset = make_balanced(mydataset(datasets_path, guidance_path, for_basemodel=False, for_testing=True, transforms=trans))
    testloader = DataLoader(testing_dset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    folders = ['bm-dm', 'bm-gan', 'bm-real']

    n_components_range = range(2, 30)

    for folder in tqdm(folders, desc="Processing folders"):
        model = backbone(backbone_name, finetuning=False, as_feature_extractor=True)
        model.eval()
        state_dict_path = os.path.join(models_dir, folder, backbone_name+folder[2:]+'.pt')
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
        
        metrics_results = calculate_metrics(data, labels, n_components_range, args.metrics)

        # Plotting
        fig, axs = plt.subplots(1, len(metrics_results), figsize=(5*len(metrics_results), 5))
        fig.suptitle(f'Clustering Metrics vs. Number of UMAP components ({folder})')

        if len(metrics_results) == 1:
            axs = [axs]

        for (metric, scores), ax in zip(metrics_results.items(), axs):
            ax.plot(n_components_range, scores, marker='o')
            ax.set_xlabel('Number of UMAP components')
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True)

        plt.tight_layout()
        save_path = f'clustering_metrics_plot_{backbone_name}_{folder}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()

if __name__ == '__main__':
    args = get_parser()
    main(args)