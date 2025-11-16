import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import glob
import seaborn as sns

sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def load_attention_maps(base_path):
    """load all attention map files in the specified path"""
    attention_maps = {}
    
    # get all .npy files
    npy_files = glob.glob(os.path.join(base_path, "*.npy"))
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        # parse file name format: attention_layer{layer}_image{image}.npy
        if filename.startswith("attention_layer") and filename.endswith(".npy"):
            parts = filename.replace(".npy", "").split("_")
            layer = int(parts[1].replace("layer", ""))
            image = int(parts[2].replace("image", ""))
            
            # load numpy array
            attention_map = np.load(file_path)
            
            if layer not in attention_maps:
                attention_maps[layer] = {}
            attention_maps[layer][image] = attention_map
    
    return attention_maps

def compute_cosine_similarity(attn1, attn2):
    """compute cosine similarity between two attention maps"""
    # flatten the attention map to a one-dimensional vector
    flat1 = attn1.flatten()
    flat2 = attn2.flatten()
    
    # compute cosine similarity
    cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
    return cos_sim


def main():
    noise_ratio = 0.3
    # define paths
    clean_path = f"wo_lora/clean/eval_epoch_20/feature_characteristics/epoch_100/attention_maps/individual_attentions"
    noise_path = f"wo_lora/noise_ratio_{noise_ratio}/eval_epoch_20/feature_characteristics/epoch_100/attention_maps/individual_attentions"

    clean_path_w = f"w_lora/lora_in_proj_initial_not_train/lora_r8/clean/eval_epoch_20/feature_characteristics/epoch_100/attention_maps/individual_attentions"
    noise_path_w = f"w_lora/lora_in_proj_initial_not_train/lora_r8/noise_ratio_{noise_ratio}/eval_epoch_20/feature_characteristics/epoch_100/attention_maps/individual_attentions"
    
    print("loading wo_lora clean attention maps...")
    clean_maps = load_attention_maps(clean_path)
    
    print("loading wo_lora noise attention maps...")
    noise_maps = load_attention_maps(noise_path)
    
    print("loading w_lora clean attention maps...")
    clean_maps_w = load_attention_maps(clean_path_w)
    
    print("loading w_lora noise attention maps...")
    noise_maps_w = load_attention_maps(noise_path_w)
    
    print(f"wo_lora Clean maps: {len(clean_maps)} layers")
    print(f"wo_lora Noise maps: {len(noise_maps)} layers")
    print(f"w_lora Clean maps: {len(clean_maps_w)} layers")
    print(f"w_lora Noise maps: {len(noise_maps_w)} layers")
    
    # compute cosine similarity for each layer - wo_lora
    layer_cos_scores = defaultdict(list)
    
    # compute cosine similarity for each layer - wo_lora
    for layer in clean_maps.keys():
        if layer in noise_maps:
            print(f"processing wo_lora layer {layer}...")
            
            # get all clean and noise images in this layer
            clean_images = list(clean_maps[layer].keys())
            noise_images = list(noise_maps[layer].keys())
            
            # for each noise image, compute cosine similarity with the corresponding clean image
            for noise_img_idx in noise_images:
                # use the noise image ID to mod the number of clean images to get the corresponding clean image
                clean_img_idx = noise_img_idx % len(clean_images)
                
                if clean_img_idx in clean_maps[layer]:
                    clean_attn = clean_maps[layer][clean_img_idx]
                    noise_attn = noise_maps[layer][noise_img_idx]
                    
                    # compute cosine similarity
                    cos_sim = compute_cosine_similarity(clean_attn, noise_attn)
                    layer_cos_scores[layer].append(cos_sim)
    
    # compute cosine similarity for each layer - w_lora
    layer_cos_scores_w = defaultdict(list)
    
    # compute cosine similarity for each layer - w_lora
    for layer in clean_maps_w.keys():
        if layer in noise_maps_w:
            print(f"processing w_lora layer {layer}...")
            
            # get all clean and noise images in this layer
            clean_images_w = list(clean_maps_w[layer].keys())
            noise_images_w = list(noise_maps_w[layer].keys())
            
            # for each noise image, compute cosine similarity with the corresponding clean image
            for noise_img_idx in noise_images_w:
                # use the noise image ID to mod the number of clean images to get the corresponding clean image
                clean_img_idx = noise_img_idx % len(clean_images_w)
                
                if clean_img_idx in clean_maps_w[layer]:
                    clean_attn = clean_maps_w[layer][clean_img_idx]
                    noise_attn = noise_maps_w[layer][noise_img_idx]
                    
                    # compute cosine similarity
                    cos_sim = compute_cosine_similarity(clean_attn, noise_attn)
                    layer_cos_scores_w[layer].append(cos_sim)
    
    # compute statistics for each layer - wo_lora
    layers = sorted(layer_cos_scores.keys())
    means = []
    stds = []
    
    for layer in layers:
        scores = layer_cos_scores[layer]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        means.append(mean_score)
        stds.append(std_score)
        
        print(f"wo_lora Layer {layer}: mean={mean_score:.4f}, std={std_score:.4f}, num_samples={len(scores)}")
    
    # compute statistics for each layer - w_lora
    layers_w = sorted(layer_cos_scores_w.keys())
    means_w = []
    stds_w = []
    
    for layer in layers_w:
        scores = layer_cos_scores_w[layer]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        means_w.append(mean_score)
        stds_w.append(std_score)
        
        print(f"w_lora Layer {layer}: mean={mean_score:.4f}, std={std_score:.4f}, num_samples={len(scores)}")
    
    # create figure
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    
    colors = {
        'wo_lora': '#2E86AB',  # dark blue
        'w_lora': '#A23B72'    # dark purple
    }
    
    # create x-axis positions
    x_positions = range(len(layers))
    layer_numbers = layers  # directly use layer numbers
    
    # plot wo_lora line chart and confidence interval
    ax.plot(x_positions, means, marker='o', label='w/o LoRA', 
            linewidth=2.5, color=colors['wo_lora'], markersize=7, 
            markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors['wo_lora'])
    ax.fill_between(x_positions, 
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.2, color=colors['wo_lora'], linewidth=0)
    
    # plot w_lora line chart and confidence interval
    if len(layers_w) == len(layers):  # ensure the same number of layers
        ax.plot(x_positions, means_w, marker='s', label='w/ LoRA', 
                linewidth=2.5, color=colors['w_lora'], markersize=7,
                markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors['w_lora'])
        ax.fill_between(x_positions, 
                        np.array(means_w) - np.array(stds_w),
                        np.array(means_w) + np.array(stds_w),
                        alpha=0.2, color=colors['w_lora'], linewidth=0)
    
    # set axis labels and title
    ax.set_xlabel('# Transformer Layer', fontweight='normal', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontweight='normal', fontsize=12)
    
    # set x-axis ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_numbers)
    
    # set y-axis range
    all_upper = np.array(means) + np.array(stds)
    all_lower = np.array(means) - np.array(stds)
    if len(layers_w) == len(layers):
        all_upper_w = np.array(means_w) + np.array(stds_w)
        all_lower_w = np.array(means_w) - np.array(stds_w)
        y_max = max(np.max(all_upper), np.max(all_upper_w))
        y_min = min(np.min(all_lower), np.min(all_lower_w))
    else:
        y_max = np.max(all_upper)
        y_min = np.min(all_lower)
    
    ax.set_ylim(y_min - 0.05, y_max + 0.05)
    
    # add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # set legend
    ax.legend(loc='lower left', frameon=False)
    
    # optimize layout
    plt.tight_layout()
    
    # save high-quality image
    os.makedirs("map_cos_plots", exist_ok=True)
    output_path = f"map_cos_plots/attention_cosine_similarity_noise_ratio_{noise_ratio}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"图表已保存到: {output_path}")
    
    # show figure
    plt.show()

if __name__ == "__main__":
    main()
