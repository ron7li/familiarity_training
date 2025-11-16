# Modeling Rapid Contextual Learning in the Visual Cortex with Fast-Weight Deep Autoencoder Networks

## Abstract
Recent neurophysiological studies have revealed that the early visual cortex can rapidly learn global image context, as evidenced by a sparsification of population responses and a reduction in mean activity when exposed to familiar versus novel image contexts. This phenomenon has been attributed primarily to local recurrent interactions, rather than changes in feedforward or feedback pathways—supported by both empirical findings and circuit-level modeling. Recurrent neural circuits capable of simulating these effects have been shown to reshape the geometry of neural manifolds, enhancing robustness and invariance to irrelevant variations. In this study, we employ a Vision Transformer (ViT)-based autoencoder to investigate, from a functional perspective, how familiarity training can induce sensitivity to global context in the early layers of a deep neural network. We hypothesize that rapid learning operates via fast weights, which encode transient or short-term memory traces, and we explore the use of Low-Rank Adaptation (LoRA) to implement such fast weights within each Transformer layer. Our results show that: (1) Familiarity training progressively increases alignment between early-layer and higher-layer latent representations, indicating growing sensitivity to global context at early stages of processing; (2) Attention maps across layers become increasingly aligned, exhibiting more coherent and consistent figure-ground awareness; (3) These effects are significantly amplified by the incorporation of LoRA-based fast weights. Together, these findings suggest that a hybrid fast-and-slow weight architecture may provide a viable computational model for studying the functional consequences of rapid global context learning in the brain.

## Environment Setup

### Create conda environment

```bash
# Create conda environment named fast_weight_familiarity
conda create -n fast_weight_familiarity python=3.10
conda activate fast_weight_familiarity

# Install dependencies
pip install -r requirements.txt
```

## Usage Workflow

### 1. Model Training

#### Train original CLIP model without LoRA
```bash
python wo_lora_train.py \
    --train_epoch 100 \
    --vit_type "ViT-B" \
    --img_patch_size 16 \
    --batch_size 4 \
    --lr 1e-4 \
    --device "cuda"
```

#### Train CLIP model with LoRA
```bash
python w_lora_train.py \
    --lora_ranks [8] \
    --train_epoch 100 \
    --vit_type "ViT-B" \
    --img_patch_size 16 \
    --batch_size 4 \
    --lr 1e-4 \
    --device "cuda"
```

### 2. Model Evaluation

#### Evaluate model without LoRA
```bash
python wo_lora_eval.py \
    --using_noise True \
    --noise_ratio 0.3 \
    --eval_epoch 20 \
    --vit_type "ViT-B" \
    --img_patch_size 16 \
    --batch_size 4 \
    --linear_lr 1e-4 \
    --linear_training_epochs 100 \
    --num_noise_pattern 5 \
    --if_attention_map True \
    --device "cuda"
```

#### Evaluate model with LoRA
```bash
python w_lora_eval.py \
    --using_noise True \
    --noise_ratio 0.3 \
    --eval_epoch 20 \
    --lora_ranks [8] \
    --vit_type "ViT-B" \
    --img_patch_size 16 \
    --batch_size 4 \
    --linear_lr 1e-4 \
    --linear_training_epochs 100 \
    --num_noise_pattern 5 \
    --if_attention_map True \
    --device "cuda"
```

**Note:** Ensure identical parameters are used when evaluating both models for fair comparison.

### 3. Attention Map Analysis

#### Attention map vs. segmentation binary mask comparison
```bash
python compare_attention_map.py 
```

This script evaluates the figure-ground overlap between attention maps generated from attention matrices and reference binary masks, comparing models with and without LoRA. The reference binary masks are obtained using [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), which combines Grounding DINO with Segment Anything Model (SAM) to provide high-quality segmentation masks for evaluation.

#### Attention map similarity computation
```bash
python compute_map_cos.py 
```

This script computes the similarity between attention maps derived from noisy images and those from clean images at the same epoch.

### 4. Manifold Transformation Analysis

#### Prepare manifold analysis data

**Without LoRA:**
```bash
python wo_lora_data_for_manifold.py 
```

**With LoRA:**
```bash
python w_lora_data_for_manifold.py 
```

#### Manifold transformation analysis
```bash
jupyter notebook manifold_analysis.ipynb
```

Perform detailed manifold transformation analysis in Jupyter notebook

## Important Notes

1. Ensure identical parameter configurations when evaluating different models
2. GPU acceleration is recommended
3. Adjust data paths according to your specific setup