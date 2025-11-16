import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from config import ids
from scipy.stats import pearsonr
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import seaborn as sns
from sklearn.cluster import SpectralClustering
import cv2
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import json
from PIL import Image
from skimage.transform import resize

from attention import generate_attention_maps


def extract_layer_features(image_encoder, images):
    """
    Extract output features from each Transformer Block in CLIP, shape is typically [seq_len, batch_size, hidden_dim].
    Returns {layer_name: tensor}.
    """
    hooks = []
    features = {}

    def register_hook(layer, name):
        def hook(module, input, output):
            features[name] = output.clone()  
        hooks.append(layer.register_forward_hook(hook))

    # Register all blocks
    for idx, block in enumerate(image_encoder.transformer.resblocks):
        register_hook(block, f"Transformer_Block_{idx}")

    with torch.no_grad():
        _ = image_encoder(images)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return features



def extract_final_block_features(image_encoder, images):
    """
    Returns the output features of the last Transformer Block in CLIP visual encoder.
    """
    last_block = image_encoder.transformer.resblocks[-1]
    container = {}

    def hook(module, input, output):
        container["final_block"] = output.clone()  

    handle = last_block.register_forward_hook(hook)
    with torch.no_grad():
        _ = image_encoder(images)
    handle.remove()

    return container["final_block"]


def train_classification_layer(
    image_encoder, 
    train_loader,   
    device,
    epochs=100,
    lr=1e-4
):
    """
    Training:
      - For each layer of image_encoder, create a linear classifier (e.g., 768->num_classes),
      - Freeze image_encoder, only train these linear layers.
    """

    sample_images, _, sample_labels = next(iter(train_loader))
    sample_images = sample_images.to(device)
    sample_labels = sample_labels.to(device)

    num_classes = len(ids)

    # Extract features once to get layer_names and hidden_dim
    image_encoder.eval()
    with torch.no_grad():
        sample_features_dict = extract_layer_features(image_encoder, sample_images)

    layer_names = list(sample_features_dict.keys())  # e.g. ["Transformer_Block_0", ..., "Transformer_Block_11"]
    # Take the output of the first layer as an example
    hidden_dim = sample_features_dict[layer_names[0]].shape[-1]

    # Create a linear classifier for each layer_name
    layer2classifier = {}
    for ln in layer_names:
        layer2classifier[ln] = nn.Linear(hidden_dim, num_classes).to(device)

    all_params = []
    for ln in layer_names:
        all_params += list(layer2classifier[ln].parameters())
    optimizer = optim.AdamW(all_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Freeze image_encoder
    image_encoder.eval()

    # Training loop
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_batches = 0

        for images, _, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Extract all layer features (no gradient)
            with torch.no_grad():
                features_dict = extract_layer_features(image_encoder, images)

            # Before training each layer's classifier, need to let them enter train mode
            for ln in layer_names:
                layer2classifier[ln].train()

            loss_per_batch = torch.tensor(0.0, device=device, requires_grad=True)

            # Iterate through each layer
            for layer_name, layer_features in features_dict.items():
                cls_token = layer_features[0, :, :]  # Take CLS

                logits = layer2classifier[layer_name](cls_token)  # [batch_size, num_classes]

                loss_layer = criterion(logits, labels)
                loss_per_batch = loss_per_batch + loss_layer

            optimizer.zero_grad()
            loss_per_batch.backward()
            optimizer.step()

            total_loss += loss_per_batch.item()
            total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        print(f"Epoch [{epoch}/{epochs}] - Avg Loss: {avg_loss:.4f}")

    return layer2classifier


def validate_classification_layer(
    image_encoder, 
    val_loader, 
    layer2classification, 
    device
):
    """
    Validate the classification performance of linear classifiers added to each layer of image_encoder
    """

    # Let image_encoder and all classifiers enter eval mode
    image_encoder.eval()
    for clf in layer2classification.values():
        clf.eval()

    # Initialize statistics
    layer_correct_counts = {ln: 0 for ln in layer2classification.keys()}
    layer_total_counts = {ln: 0 for ln in layer2classification.keys()}
    layer_correct_conf_sum = {ln: 0.0 for ln in layer2classification.keys()}

    # Iterate through validation set
    with torch.no_grad():
        for _, images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Extract features of all layers
            features_dict = extract_layer_features(image_encoder, images)
            
            # Predict for each layer's linear classifier
            for layer_name, layer_features in features_dict.items():
                cls_token = layer_features[0, :, :]

                logits = layer2classification[layer_name](cls_token)
                probs = torch.softmax(logits, dim=-1)  # Get prediction probability distribution

                # Calculate predicted labels
                pred_labels = probs.argmax(dim=-1)  # [batch_size]

                # Count correct predictions
                correct_mask = (pred_labels == labels)  # bool tensor
                layer_correct_counts[layer_name] += correct_mask.sum().item()
                layer_total_counts[layer_name] += batch_size

                # Count the prediction probability (confidence) for the true class
                correct_label_prob = probs[torch.arange(batch_size), labels]
                layer_correct_conf_sum[layer_name] += correct_label_prob.sum().item()

    # Calculate final accuracy and average confidence
    layer_acc_dict = {}
    layer_prob_dict = {}
    for layer_name in layer2classification.keys():
        total = layer_total_counts[layer_name]
        if total == 0:
            layer_acc_dict[layer_name] = 0.0
            layer_prob_dict[layer_name] = 0.0
        else:
            correct = layer_correct_counts[layer_name]
            acc = correct / total
            avg_prob = layer_correct_conf_sum[layer_name] / total

            layer_acc_dict[layer_name] = acc
            layer_prob_dict[layer_name] = avg_prob

    return layer_acc_dict, layer_prob_dict


def plot_feature_distribution(layer_preds_mean_dict, reals_mean, save_dir, plot_all=True):
    """
    Plot feature distribution
    """
    os.makedirs(save_dir, exist_ok=True)

    # Calculate y-axis range
    all_values = []
    for preds_mean in layer_preds_mean_dict.values():
        all_values.extend(preds_mean)
    all_values.extend(reals_mean)
    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05

    if plot_all:
        plt.figure(figsize=(15, 8))
        for layer_name, preds_mean in layer_preds_mean_dict.items():
            plt.plot(preds_mean, label=f'{layer_name}_noise', alpha=0.7)
        plt.plot(reals_mean, label=f'Transformer_Block_11_clean', alpha=0.7, linestyle='--')
        plt.title('Feature Distribution Comparison Across Layers')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.ylim(y_min, y_max)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_distribution_all.png'))
        plt.close()

    # Plot comparison for each layer
    for layer_name, preds_mean in layer_preds_mean_dict.items():
        plt.figure(figsize=(15, 8))
        plt.plot(preds_mean, label=f'{layer_name}_noise', alpha=0.7)
        plt.plot(reals_mean, label=f'Transformer_Block_11_clean', alpha=0.7, linestyle='--')
        plt.title('Feature Distribution Comparison Across Layers')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.ylim(y_min, y_max)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'feature_distribution_{layer_name}.png'))
        plt.close()

def plot_feature_tsne(layer_preds_mean_dict, reals_mean, save_dir, plot_all=True):
    """
    Use t-SNE to reduce the dimension and visualize the features
    """
    os.makedirs(save_dir, exist_ok=True)

    all_features = []
    labels = []
    for layer_name, preds_mean in layer_preds_mean_dict.items():
        all_features.append(preds_mean)
        labels.append(f'{layer_name}_noise')
    all_features.append(reals_mean)
    labels.append('clean')
    
    all_features = np.array(all_features)
    
    n_samples = len(all_features)
    # Set perplexity to half of the sample size, but not exceeding 30
    perplexity = min(n_samples - 1, 30)
    
    # Use t-SNE to reduce the dimension
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(all_features)

    if plot_all:
        plt.figure(figsize=(10, 8))
        # Plot all points
        for i, label in enumerate(labels):
            if label == 'clean':
                plt.scatter(features_2d[i, 0], features_2d[i, 1], label=label, marker='*', s=200)
            else:
                plt.scatter(features_2d[i, 0], features_2d[i, 1], label=label, alpha=0.7)
        
        plt.title('t-SNE Visualization of Features Across Layers')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_tsne_all.png'))
        plt.close()

    # Plot comparison for each layer
    else:
        for i, (layer_name, _) in enumerate(layer_preds_mean_dict.items()):
            plt.figure(figsize=(10, 8))
            plt.scatter(features_2d[i, 0], features_2d[i, 1], label=f'{layer_name}_noise', alpha=0.7)
            plt.scatter(features_2d[-1, 0], features_2d[-1, 1], label='clean', marker='*', s=200)
            
        plt.title(f't-SNE Visualization of Features for {layer_name}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'feature_tsne_{layer_name}.png'))
        plt.close()

def plot_feature_visualizations(layer_preds_mean_dict, reals_mean, save_dir):
    """
    Use multiple methods to visualize the feature distribution
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_features = []
    labels = []
    for layer_name, preds_mean in layer_preds_mean_dict.items():
        all_features.append(preds_mean)
        labels.append(f'{layer_name}_noise')
    all_features.append(reals_mean)
    labels.append('clean')
    all_features = np.array(all_features)
    
    # 1. PCA visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        if label == 'clean':
            plt.scatter(features_2d[i, 0], features_2d[i, 1], label=label, marker='*', s=200)
        else:
            plt.scatter(features_2d[i, 0], features_2d[i, 1], label=label, alpha=0.7)
    plt.title('PCA Visualization of Features')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_pca.png'))
    plt.close()

    
    # 2. Heatmap visualization
    plt.figure(figsize=(15, 8))   
    min_max_features = (all_features - all_features.min(axis=1, keepdims=True)) / (all_features.max(axis=1, keepdims=True) - all_features.min(axis=1, keepdims=True) + 1e-9)
    plt.imshow(min_max_features, aspect='auto', cmap='RdBu_r')
    plt.colorbar(label='Min-Max Normalized Value')
    plt.title('Heatmap (Min-Max)')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Layer')
    plt.yticks(range(len(labels)), labels)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_heatmap.png'))
    plt.close()
    
    # 3. Boxplot visualization
    plt.figure(figsize=(15, 8))
    data_to_plot = []
    for i, label in enumerate(labels):
        data_to_plot.append(all_features[i])
    box_plot = plt.boxplot(data_to_plot)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title('Feature Distribution Box Plot')
    plt.xlabel('Layer')
    plt.ylabel('Feature Value')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_boxplot.png'))
    plt.close()

def plot_feature_correlation(layer_dimension_corr_dict, save_dir):
    """
    Plot the feature dimension correlation heatmap
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有层名
    layer_names = list(layer_dimension_corr_dict.keys())

    vmin = min([np.min(corr_matrix) for corr_matrix in layer_dimension_corr_dict.values()])

    for layer_name in layer_names:
        dim_corr = layer_dimension_corr_dict[layer_name]
        plt.figure(figsize=(10, 8))
        sns.heatmap(dim_corr, cmap='RdBu_r', center=0, square=True, xticklabels=False, yticklabels=False, vmin=vmin, vmax=1, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f'Feature Dimension Correlation for {layer_name}')
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'feature_correlation_{layer_name}.png'))
        plt.close()

def calculate_attention_mask_iou(attn_maps, save_dir, layer_names):
    """
    Calculate the IoU of attention maps and masks
    """
    layer_ious = {layer_name: [] for layer_name in layer_names}
    
    if save_dir and attn_maps:
        print("Calculating the IoU of attention maps and masks...")
        
        mask_dir = 'segmentation_masks'
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory {mask_dir} not found. Skipping IOU calculation.")
            return {}
        
        # Calculate the IoU of each attention map
        for layer_name, attn_maps_list in attn_maps.items():
            for img_idx, (attn_map, label) in enumerate(attn_maps_list):
                # Build the mask path (assuming the mask file name format is mask_{img_idx}.jpg)
                mask_path = os.path.join(mask_dir, f'mask_{label}.jpg')
                
                if os.path.exists(mask_path):
                    mask = load_mask(mask_path)
                    if mask is not None:
                        iou = calculate_iou(attn_map, mask)
                        layer_ious[layer_name].append(iou)
                else:
                    print(f"Warning: Mask file {mask_path} not found for image {img_idx}")
        
        # Calculate the IoU statistics of each layer
        layer_iou_stats = {}
        for layer_name, iou_list in layer_ious.items():
            if iou_list:
                mean_iou = np.mean(iou_list)
                std_iou = np.std(iou_list)
                layer_iou_stats[layer_name] = {
                    'mean_iou': float(mean_iou),
                    'std_iou': float(std_iou),
                    'num_samples': len(iou_list)
                }
            else:
                layer_iou_stats[layer_name] = {
                    'mean_iou': 0.0,
                    'std_iou': 0.0,
                    'num_samples': 0
                }
        
        # Save the IoU statistics to a JSON file
        iou_save_path = os.path.join(save_dir, 'attention_mask_iou_stats.json')
        with open(iou_save_path, 'w', encoding='utf-8') as f:
            json.dump(layer_iou_stats, f, indent=2, ensure_ascii=False)
        
        print(f"The IoU statistics have been saved to: {iou_save_path}")
        
        # Print the IoU statistics summary
        print("\n=== Attention Map vs Mask IOU Statistics Summary ===")
        for layer_name, stats in layer_iou_stats.items():
            print(f"{layer_name}: IOU = {stats['mean_iou']:.3f} ± {stats['std_iou']:.3f} (n={stats['num_samples']})")
        
        return layer_iou_stats
    
    return {}


def calculate_iou(attn_map, mask, threshold=0.5, already_normalized=True):
    """
    Calculate the IoU of single attention map and mask
    """
    # If the attention map is already normalized, use it directly
    if already_normalized:
        attn_norm = attn_map
    else:
        # If not normalized, normalize it (this should be very rare)
        min_attn = attn_map.min()
        max_attn = attn_map.max()
        attn_norm = (attn_map - min_attn) / (max_attn - min_attn + 1e-8)
    
    # Resize the attention map to match the mask
    if attn_norm.shape != mask.shape:
        print(f"Attention map shape: {attn_norm.shape}, mask shape: {mask.shape}")
        attn_norm = resize(attn_norm, mask.shape, order=1, mode='reflect', anti_aliasing=True)
    
    # Binarize the attention map
    attn_bin = (attn_norm > threshold).astype(np.uint8)
    
    # Calculate the IoU of the attention map and the mask, and the IoU of the attention map and the inverse mask, and take the maximum value
    intersection1 = np.logical_and(mask, attn_bin).sum()
    union1 = np.logical_or(mask, attn_bin).sum()
    iou1 = intersection1 / union1 if union1 > 0 else 0
    
    mask_inv = 1 - mask
    intersection2 = np.logical_and(mask_inv, attn_bin).sum()
    union2 = np.logical_or(mask_inv, attn_bin).sum()
    iou2 = intersection2 / union2 if union2 > 0 else 0
    
    iou = max(iou1, iou2)
    return iou


def load_mask(mask_path, target_size=(224, 224)):
    """
    Load and preprocess the mask
    """
    try:
        mask_img = Image.open(mask_path).resize(target_size).convert('L')
        mask = np.array(mask_img)
        mask = (mask > 128).astype(np.uint8)
        return mask
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
        return None


def validate_layer_features(
    image_encoder,               
    val_loader,                      
    device, 
    visualize=False,
    save_dir=None,
    using_noise=True,
    validate_correlation=False,
    use_attention_maps=True,
    if_save_attention_maps=True,
):

    image_encoder.eval()
    
    # get all the layer names
    sample_images_a, sample_images_b, sample_labels = next(iter(val_loader))
    sample_images_a = sample_images_a.to(device)
    sample_images_b = sample_images_b.to(device)
    sample_labels = sample_labels.to(device)

    with torch.no_grad():
        sample_features_dict_a = extract_layer_features(image_encoder, sample_images_a)

    layer_names = list(sample_features_dict_a.keys())  # e.g. ["Transformer_Block_0", ..., "Transformer_Block_11"]

    if use_attention_maps:
        attn_maps = generate_attention_maps(image_encoder, val_loader, if_save=if_save_attention_maps, save_dir=save_dir, device=device)
        
        if save_dir:
            calculate_attention_mask_iou(attn_maps, save_dir, layer_names)

    # collect the predictions and real values
    layer_preds = {ln: [] for ln in layer_names}
    layer_reals = {ln: [] for ln in layer_names}

    if using_noise and visualize:
        layer_noise_preds = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch_idx, (images_clean, images_noise, sample_labels) in enumerate(val_loader):
            images_clean = images_clean.to(device)
            images_noise = images_noise.to(device)
            sample_labels = sample_labels.to(device)

            # Extract the features of all layers of noise
            features_dict_noise = extract_layer_features(image_encoder, images_noise)

            # Extract the final layer feature of clean
            final_features_clean = extract_final_block_features(image_encoder, images_clean) # [seq_len, batch_size, 768]
            cls_token_clean_final = final_features_clean[0, :, :]                          # [batch_size, 768]

            real_vec = cls_token_clean_final / (cls_token_clean_final.norm(dim=-1, keepdim=True) + 1e-9)

            for layer_name, layer_features_noise in features_dict_noise.items():
                cls_token_noise = layer_features_noise[0, :, :]  # [batch_size, 768]
                
                pred_vec = cls_token_noise

                # Normalize
                pred_vec = pred_vec / (pred_vec.norm(dim=-1, keepdim=True) + 1e-9)

                layer_preds[layer_name].append(pred_vec.cpu().numpy())
                layer_reals[layer_name].append(real_vec.cpu().numpy())

                if using_noise and visualize:
                    for idx, label in enumerate(sample_labels.cpu().numpy()):
                        layer_noise_preds[label][layer_name].append(pred_vec[idx].cpu().numpy())
    

    # Calculate the metrics of each layer
    layer_cos_dict = {}
    layer_pcc_dict = {}
    if visualize:
        layer_preds_sorted_mean_dict = {}
        layer_preds_mean_dict = {}
        layer_sorted_dimension_corr_dict = {}
        layer_sorted_dimension_corr_dict_cluster = {}
    
    for layer_name in layer_names:
        preds_np = np.vstack(layer_preds[layer_name])  # shape [num_samples, 768]
        reals_np = np.vstack(layer_reals[layer_name])  # shape [num_samples, 768]

        # calculate cosine similarity
        preds_tensor = torch.from_numpy(preds_np).float()
        reals_tensor = torch.from_numpy(reals_np).float()
        
        # Calculate the cosine similarity
        cos_each = F.cosine_similarity(preds_tensor, reals_tensor, dim=1)
        layer_cos_dict[layer_name] = float(cos_each.mean())
        
        # calculate PCC
        preds_sorted = -np.sort(-preds_np, axis=1)
        reals_sorted = -np.sort(-reals_np, axis=1)
        
        num_samples = preds_np.shape[0]
        pcc_values = []
        for i in range(num_samples):
            pcc, _ = pearsonr(preds_sorted[i], reals_sorted[i])
            pcc_values.append(pcc)
        layer_pcc_dict[layer_name] = float(np.mean(pcc_values))

        if visualize:
            preds_sorted_mean = np.mean(preds_sorted, axis=0)  
            layer_preds_sorted_mean_dict[layer_name] = preds_sorted_mean
            preds_mean = np.mean(preds_np, axis=0)
            layer_preds_mean_dict[layer_name] = preds_mean

            # calculate dimension correlation
            if validate_correlation:
                if not using_noise:
                    dim_corr = np.corrcoef(preds_sorted.T)
                    layer_sorted_dimension_corr_dict[layer_name] = dim_corr
                    # spectral clustering
                    layer_sorted_dimension_corr_dict_cluster[layer_name] = spectral_clustering_dim_correlation(dim_corr)
                else:
                    noise_dim_corrs = []
                    for label in layer_noise_preds.keys():
                        noise_preds_sorted = -np.sort(-np.vstack(layer_noise_preds[label][layer_name]), axis=1)
                        dim_corr = np.corrcoef(noise_preds_sorted.T)  # shape [768, 768]
                        noise_dim_corrs.append(dim_corr)
                        
                    dim_corr_mean = np.mean(noise_dim_corrs, axis=0)
                    layer_sorted_dimension_corr_dict[layer_name] = dim_corr_mean
                    # spectral clustering
                    layer_sorted_dimension_corr_dict_cluster[layer_name] = spectral_clustering_dim_correlation(dim_corr_mean)

    if visualize and save_dir:
        reals_sorted_mean = np.mean(reals_sorted, axis=0) 
        plot_feature_distribution(layer_preds_sorted_mean_dict, reals_sorted_mean, os.path.join(save_dir, 'sorted'), plot_all=True) 
        reals_mean = np.mean(reals_np, axis=0)
        plot_feature_distribution(layer_preds_mean_dict, reals_mean, os.path.join(save_dir, 'unsorted'), plot_all=True)
        
        # plot_feature_visualizations(layer_preds_mean_dict, reals_mean, os.path.join(save_dir, 'visualizations'))

    if validate_correlation and save_dir:
        plot_feature_correlation(layer_sorted_dimension_corr_dict, os.path.join(save_dir, 'dim_correlation'))
        # plot_clustered_correlation(layer_sorted_dimension_corr_dict, layer_sorted_dimension_corr_dict_cluster, os.path.join(save_dir, 'dim_correlation_clustered'))
        return layer_cos_dict, layer_pcc_dict, layer_sorted_dimension_corr_dict, layer_sorted_dimension_corr_dict_cluster

    return layer_cos_dict, layer_pcc_dict


def spectral_clustering_dim_correlation(dim_corr, n_clusters=10, affinity='precomputed', assign_labels='kmeans', random_state=42):#
    dim_corr_nonneg = (dim_corr + 1) / 2
    clustering_model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, assign_labels=assign_labels, random_state=random_state)
    dim_cluster_labels = clustering_model.fit_predict(dim_corr_nonneg)
    return dim_cluster_labels


def plot_clustered_correlation(corr_matrix_dict, cluster_labels_dict, save_dir):
    """
    Plot the clustered correlation matrix
    """
    os.makedirs(save_dir, exist_ok=True)

    vmin = min([np.min(corr_matrix) for corr_matrix in corr_matrix_dict.values()])

    for layer_name, corr_matrix in corr_matrix_dict.items():
        cluster_labels = cluster_labels_dict[layer_name]
        sorted_indices = np.argsort(cluster_labels)
        sorted_corr = corr_matrix[sorted_indices][:, sorted_indices]

        plt.figure(figsize=(10, 8))
        sns.heatmap(sorted_corr, cmap='RdBu_r', center=0, square=True, xticklabels=False, yticklabels=False, vmin=vmin, vmax=1, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title(f"Dimension Correlation Matrix Sorted by Cluster for {layer_name}")
        plt.xlabel('Dimension')
        plt.ylabel('Dimension')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'clustered_correlation_matrix_{layer_name}.png'))
        plt.close()


def plot_dicts_subplots(output_dicts, checkpoints, save_path, mode):
    """
    Plot 5 subplots and save the image to the local (save_path).
    """
    layer_names = list(output_dicts[0].keys())
    layer_ids = list(range(len(layer_names)))

    fig, axes = plt.subplots(1, 6, figsize=(20, 4), sharey=True)
    axes = axes.flatten()  # flatten 

    # Calculate the minimum value of all data as the lower limit
    min_value = float('inf')
    for mse_dict in output_dicts:
        min_value = min(min_value, min(mse_dict.values()))
    if min_value < 0:
        min_value -= 0.05
    else:
        min_value = 0

    for i, (mse_dict, epoch) in enumerate(zip(output_dicts, checkpoints)):
        ax = axes[i]
        mse_values = [mse_dict[layer] for layer in layer_names]

        # Plot the bar chart
        ax.bar(layer_ids, mse_values, color='skyblue')
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlabel("# Transformer Layer")
    
        ax.set_ylabel(mode)
        if mode != "Correct Class Prediction Probability":
            ax.set_ylim(bottom=min_value, top=1.05) 


        ax.set_xticks(layer_ids)
        ax.set_xticklabels(layer_ids, rotation=0)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_single_plot(output_dicts, checkpoints, save_path, mode):
    layer_names = list(output_dicts[0].keys())
    layer_ids = list(range(len(layer_names)))

    mse_values = [output_dicts[0][layer] for layer in layer_names]
    plt.bar(layer_ids, mse_values, color='skyblue')
    plt.title("Epoch 0")
    plt.xlabel("# Transformer Layer")
    plt.ylabel(mode)
    plt.xticks(layer_ids, [str(i) for i in layer_ids], rotation=0)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()