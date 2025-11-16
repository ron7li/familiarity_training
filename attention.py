import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def extract_attention_weights(image_encoder, images):
    """
    Extract the attention weights of each Transformer Block in CLIP visual encoder
    Returns {layer_name: attention_weights}
    """
    hooks = []
    attention_weights = {}
    
    def create_hook(layer_name):
        # This hook is registered on the ResidualAttentionBlock
        def hook(module, input, output):
            # The input to the attention layer is the output of the first LayerNorm.
            x_norm = module.ln_1(input[0])
            
            # Call the attention submodule directly to avoid recursion and get the averaged attention weights.
            with torch.no_grad():
                # module.attn is nn.MultiheadAttention.
                # It returns averaged weights by default.
                _, attn = module.attn(x_norm, x_norm, x_norm, need_weights=True)
                attention_weights[layer_name] = attn.clone()  
        return hook
    
    # Register a forward hook on each ResidualAttentionBlock
    for idx, block in enumerate(image_encoder.transformer.resblocks):
        layer_name = f"Transformer_Block_{idx}"
        hook = block.register_forward_hook(create_hook(layer_name))
        hooks.append(hook)
    
    # The forward pass will trigger the hooks
    with torch.no_grad():
        _ = image_encoder(images)
    
    # remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights


def number_to_letter(num):
    """Convert number to letter (0->A, 1->B, 2->C, ...)"""
    return chr(65 + num)  


def generate_attention_maps(image_encoder, val_loader, if_save=False, save_dir=None, device='cuda'):
    """
    generate attention maps from attention weights and save them
    
    Args:
    - image_encoder: CLIP visual encoder
    - val_loader: data loader, each iteration returns (images_clean, images_noise, labels)
    - if_save: whether to save the results
    - save_dir: save directory
    - device: device
    
    Returns:
    - attn_maps: dictionary format {layer_name: [(attention_map, label), ...]}

    Note:
    - Use images_noise as input
    - Normalize each layer's maps using the max and min values of the maps in this layer
    """
    image_encoder.eval()
    
    if if_save and save_dir is not None:
        attention_dir = os.path.join(save_dir, 'attention_maps')
        os.makedirs(attention_dir, exist_ok=True)
        
        individual_dir = os.path.join(attention_dir, 'individual_attentions')
        os.makedirs(individual_dir, exist_ok=True)
    
    patch_size = image_encoder.conv1.kernel_size[0]
    input_resolution = image_encoder.input_resolution
    grid_size = input_resolution // patch_size
    num_layers = len(image_encoder.transformer.resblocks)
    
    all_images = []
    all_labels = []
    
    print("Collecting data from val_loader...")
    
    if len(val_loader) == 0:
        raise ValueError("val_loader is empty, cannot generate attention maps")
    
    for batch_idx, (images_clean, images_noise, labels) in enumerate(val_loader):
        all_images.append(images_noise.to(device))
        all_labels.append(labels.to(device))
        print(f"Processing batch {batch_idx + 1}/{len(val_loader)}")
    
    if len(all_images) == 0:
        raise ValueError("No data collected from val_loader")

    batch_size = all_images[0].size(0)
    
    # merge all batches
    images_to_plot = torch.cat(all_images, dim=0)
    labels_to_plot = torch.cat(all_labels, dim=0)
    num_images = images_to_plot.size(0)
    
    print(f"Collected {num_images} images")
    
    # extract attention weights
    attention_weights = extract_attention_weights(image_encoder, images_to_plot)
    
    # calculate the max and min of each layer
    layer_min_max = {}
    
    for layer_name in attention_weights.keys():
        attn_for_layer = attention_weights[layer_name]  
        # print(attn_for_layer.shape)
        layer_min = float('inf')
        layer_max = float('-inf')
        
        for img_idx in range(num_images):
            cls_attention = attn_for_layer[img_idx, 0, 1:]  # [grid_size^2]
            attention_map_2d = cls_attention.view(grid_size, grid_size)
            attention_map_np = attention_map_2d.cpu().numpy()
            layer_min = min(layer_min, attention_map_np.min())
            layer_max = max(layer_max, attention_map_np.max())
        
        layer_min_max[layer_name] = (layer_min, layer_max)
        print(f"{layer_name}: [{layer_min:.6f}, {layer_max:.6f}]")
    
    # save each attention map separately
    attn_maps = {layer_name: [] for layer_name in attention_weights.keys()}

    for layer_idx in range(num_layers):
        layer_name = f'Transformer_Block_{layer_idx}'
        
        if layer_name in attention_weights:
            attn_for_layer = attention_weights[layer_name]  
            layer_min, layer_max = layer_min_max[layer_name]
            
            for img_idx, label in zip(range(num_images), labels_to_plot):
                # Get CLS attention, reshape, and resize
                cls_attention = attn_for_layer[img_idx, 0, 1:]  # [grid_size^2]
                attention_map_2d = cls_attention.view(grid_size, grid_size)
                attention_map_np = attention_map_2d.cpu().numpy()
                
                # normalize using the max and min of this layer
                attention_map_np = (attention_map_np - layer_min) / (layer_max - layer_min + 1e-8)
                attention_map_resized = cv2.resize(attention_map_np, (input_resolution, input_resolution), interpolation=cv2.INTER_CUBIC)
                attn_maps[layer_name].append((attention_map_resized, label))
                
                if if_save:
                    # save numpy file (keep the original values)
                    npy_filename = f'attention_layer{layer_idx}_image{img_idx}.npy'
                    npy_path = os.path.join(individual_dir, npy_filename)
                    np.save(npy_path, attention_map_resized)

    if if_save:
        print(f"Individual attention maps of first batch saved to: {individual_dir}")
        print(f"Saved {num_layers} layers x {num_images} images = {num_layers * num_images} attention maps")

    if if_save:
        # create grid plot (only display the first 4 images)
        display_images = min(4, num_images)
        images_for_grid = images_to_plot[:display_images]
        
        # create a large canvas: display_images rows, 13 columns
        fig, axes = plt.subplots(display_images, num_layers + 1, figsize=(2 * (num_layers + 1), 2 * display_images))
        if display_images == 1:
            axes = axes.reshape(1, -1) # Ensure axes is 2D even for 1 image

        for img_idx in range(display_images):
            # --- Column 0: Plot Original Image ---
            ax = axes[img_idx, 0]
            original_image = images_for_grid[img_idx].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = np.clip(original_image * std + mean, 0, 1)
            
            ax.imshow(original_image)

            # add title on the left
            letter_label = number_to_letter(img_idx)
            ax.set_ylabel(f'Input {letter_label}', rotation=0, labelpad=40, fontsize=12, va='center')
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
            # --- Columns 1-12: Plot Attention Maps ---
            for layer_idx in range(num_layers):
                ax = axes[img_idx, layer_idx + 1]
                layer_name = f'Transformer_Block_{layer_idx}'
                
                if layer_name in attention_weights:
                    attn_for_layer = attention_weights[layer_name]  
                    layer_min, layer_max = layer_min_max[layer_name]
                    
                    # Get CLS attention, reshape, and resize
                    cls_attention = attn_for_layer[img_idx, 0, 1:]  # [grid_size^2]
                    attention_map_2d = cls_attention.view(grid_size, grid_size)
                    attention_map_np = attention_map_2d.cpu().numpy()
                    
                    # normalize using the max and min of this layer
                    attention_map_np = (attention_map_np - layer_min) / (layer_max - layer_min + 1e-8)
                    attention_map_resized = cv2.resize(attention_map_np, (input_resolution, input_resolution), interpolation=cv2.INTER_CUBIC)
                    
                    # use a uniform color mapping range [0, 1]
                    im = ax.imshow(attention_map_resized, cmap='jet', vmin=0, vmax=1)
                
                if img_idx == 0:
                    ax.set_title(f'Layer {layer_idx}')
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
        
        # save the whole grid plot
        save_path = os.path.join(attention_dir, 'attention_maps_grid.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Attention maps grid saved to: {save_path}")
    
    return attn_maps

