import os
import json
import numpy as np
import matplotlib.pyplot as plt

noise_dir = 'noise_ratio_0.3'
eval_epoch = 20
train_epoch = 100
lora_rank = 8

wo_lora_json = f"wo_lora/{noise_dir}/eval_epoch_{eval_epoch}/feature_characteristics/epoch_{train_epoch}/attention_mask_iou_stats.json"
w_lora_json = f"w_lora/lora_in_proj_initial_not_train/lora_r{lora_rank}/{noise_dir}/eval_epoch_{eval_epoch}/feature_characteristics/epoch_{train_epoch}/attention_mask_iou_stats.json"

# load IoU stats from JSON file
def load_iou_stats(json_path):
    """load IoU stats from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"file not found: {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"JSON decode error: {json_path}")
        return {}

# load data
wo_lora_stats = load_iou_stats(wo_lora_json)
w_lora_stats = load_iou_stats(w_lora_json)

if not wo_lora_stats or not w_lora_stats:
    print("failed to load data, please check the file path")
    exit()

# extract layer names and stats data
layers = list(wo_lora_stats.keys())
layers.sort(key=lambda x: int(x.split('_')[-1]))  # 按层号排序

# extract mean and std data
wo_lora_means = [wo_lora_stats[layer]['mean_iou'] for layer in layers]
wo_lora_stds = [wo_lora_stats[layer]['std_iou'] for layer in layers]
w_lora_means = [w_lora_stats[layer]['mean_iou'] for layer in layers]
w_lora_stds = [w_lora_stats[layer]['std_iou'] for layer in layers]

# plot line chart
os.makedirs(f'attention_map_analysis/{noise_dir}', exist_ok=True)

import seaborn as sns
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# set font and style
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

# create figure
fig, ax = plt.subplots(figsize=(6.5, 4.0)) 

colors = {
    'wo_lora': '#2E86AB',  # dark blue
    'w_lora': '#A23B72'    # dark purple
}

# create x-axis positions
x_positions = range(len(layers))
layer_numbers = [int(layer.split('_')[-1]) for layer in layers]  # extract layer number

# plot wo_lora line chart and confidence interval
ax.plot(x_positions, wo_lora_means, marker='o', label='w/o LoRA', 
        linewidth=2.5, color=colors['wo_lora'], markersize=7, 
        markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors['wo_lora'])
ax.fill_between(x_positions, 
                np.array(wo_lora_means) - np.array(wo_lora_stds),
                np.array(wo_lora_means) + np.array(wo_lora_stds),
                alpha=0.2, color=colors['wo_lora'], linewidth=0)

# plot w_lora line chart and confidence interval
ax.plot(x_positions, w_lora_means, marker='s', label='w/ LoRA', 
        linewidth=2.5, color=colors['w_lora'], markersize=7,
        markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors['w_lora'])
ax.fill_between(x_positions, 
                np.array(w_lora_means) - np.array(w_lora_stds),
                np.array(w_lora_means) + np.array(w_lora_stds),
                alpha=0.2, color=colors['w_lora'], linewidth=0)

# set axis labels and title
ax.set_xlabel('# Transformer Layer', fontweight='normal', fontsize=12)
ax.set_ylabel('Average fg-IoU', fontweight='normal', fontsize=12)
# ax.set_title('Layer-wise Figure-Ground IoU Comparison', fontweight='normal', fontsize=13, pad=15)

# set x-axis ticks
ax.set_xticks(x_positions)
ax.set_xticklabels(layer_numbers)

# set y-axis range, leave appropriate blank
# calculate the max and min of all mean±std
all_upper = np.array(wo_lora_means) + np.array(wo_lora_stds)
all_lower = np.array(wo_lora_means) - np.array(wo_lora_stds)
all_upper_w = np.array(w_lora_means) + np.array(w_lora_stds)
all_lower_w = np.array(w_lora_means) - np.array(w_lora_stds)

y_max = max(np.max(all_upper), np.max(all_upper_w))
y_min = min(np.min(all_lower), np.min(all_lower_w))

ax.set_ylim(y_min - 0.05, y_max + 0.05)

# add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# set legend
ax.legend(loc='upper left', frameon=False)

# optimize layout
plt.tight_layout()

# save high-quality image
save_path = f'attention_map_analysis/{noise_dir}/iou_comparison_rank{lora_rank}_ep{train_epoch}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# print stats summary
print("=" * 60)
print("IoU COMPARISON ANALYSIS")
print("=" * 60)
print(f"Dataset: {noise_dir}")
print(f"Evaluation Epoch: {eval_epoch}")
print(f"Training Epoch: {train_epoch}")
print(f"LoRA Rank: {lora_rank}")
print()

print("OVERALL PERFORMANCE:")
print("-" * 30)
print(f"w/o LoRA: {np.mean(wo_lora_means):.3f} ± {np.mean(wo_lora_stds):.3f}")
print(f"w/  LoRA: {np.mean(w_lora_means):.3f} ± {np.mean(w_lora_stds):.3f}")
improvement = np.mean(w_lora_means) - np.mean(wo_lora_means)
improvement_pct = improvement/np.mean(wo_lora_means)*100
print(f"Improvement: {improvement:.3f} ({improvement_pct:+.1f}%)")
print()

print("LAYER-WISE PERFORMANCE:")
print("-" * 50)
print(f"{'Layer':<8} {'w/o LoRA':<12} {'w/ LoRA':<12} {'Improvement':<12}")
print("-" * 50)
for i, layer in enumerate(layers):
    layer_num = layer_numbers[i]
    wo_mean = wo_lora_means[i]
    w_mean = w_lora_means[i]
    layer_improvement = w_mean - wo_mean
    print(f"{layer_num:<8} {wo_mean:<12.3f} {w_mean:<12.3f} {layer_improvement:<+12.3f}")

print("-" * 50)
print(f"Chart saved to: {save_path}")
print("=" * 60)







