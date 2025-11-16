import torch
import clip
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
import os
import random
import numpy as np
from dataset import CustomDataset
import torch.nn as nn
from utils import *
from config import ids
from decoder import TransformerDecoder


def get_data_at_level_and_epoch(image_encoder, val_loader, device, using_noise=True):
    features_in_dict = {}
    features_out_dict = {}
    image_encoder.eval()

    hooks = []
    layer_inputs = {}
    layer_outputs = {}
    
    def create_hook(layer_name):
        def hook_fn(module, input, output):
            # save CLS token of input feature (input is a tuple, take the first element's 0th token)
            layer_inputs[layer_name] = input[0][0, :, :].clone().detach()  # [batch_size, hidden_dim]
            # save CLS token of output feature
            layer_outputs[layer_name] = output[0, :, :].clone().detach()  # [batch_size, hidden_dim]
        return hook_fn
    
    # register hook for each transformer block
    for idx, block in enumerate(image_encoder.transformer.resblocks):
        layer_name = f"transformer_block_{idx}"
        hook = block.register_forward_hook(create_hook(layer_name))
        hooks.append(hook)

    with torch.no_grad():
        # collect features of all batches
        all_inputs = {f"transformer_block_{idx}": [] for idx in range(len(image_encoder.transformer.resblocks))}
        all_outputs = {f"transformer_block_{idx}": [] for idx in range(len(image_encoder.transformer.resblocks))}
        all_labels = []
        
        for images_clean, images_noise, sample_labels in val_loader:
            images_clean = images_clean.to(device)
            images_noise = images_noise.to(device)
            
            # process noisy images
            if using_noise:
                _ = image_encoder(images_noise)
            else:
                _ = image_encoder(images_clean)
            
            # collect CLS token features of this batch
            for layer_name in layer_inputs:
                all_inputs[layer_name].append(layer_inputs[layer_name])
                all_outputs[layer_name].append(layer_outputs[layer_name])
            
            all_labels.extend(sample_labels.tolist())
    
    # remove hooks
    for hook in hooks:
        hook.remove()
    
    # organize features to (n_label, n_img_of_label, d) format
    num_classes = len(set(all_labels))  # number of classes
    imgs_per_class = len(all_labels) // num_classes  # number of images per class
    
    for layer_name in all_inputs:
        # concatenate CLS token features of all batches
        input_features = torch.cat(all_inputs[layer_name], dim=0)  # [total_imgs, hidden_dim]
        output_features = torch.cat(all_outputs[layer_name], dim=0)  # [total_imgs, hidden_dim]
        
        # reorganize to (n_label, n_img_of_label, hidden_dim) format
        hidden_dim = input_features.shape[1]
        
        # group by label - cannot use view directly, need to reorder by label
        input_by_class = torch.zeros(num_classes, imgs_per_class, hidden_dim)
        output_by_class = torch.zeros(num_classes, imgs_per_class, hidden_dim)
        
        # count number of images already put into each class
        class_counts = [0] * num_classes
        
        # put features in the correct position by label order
        for i, label in enumerate(all_labels):
            input_by_class[label, class_counts[label]] = input_features[i]
            output_by_class[label, class_counts[label]] = output_features[i]
            class_counts[label] += 1
        
        features_in_dict[layer_name] = input_by_class
        features_out_dict[layer_name] = output_by_class
    
    return features_in_dict, features_out_dict


def main(noise_levels = [0,0.1, 0.3, 0.5],
         eval_epoch = 2,
         vit_type = "ViT-B",
         img_patch_size = 16,
         batch_size = 4,
         dir = "wo_lora",
         device = "cuda"):

    # create a dictionary to save features of all noise levels
    all_noise_features_in = {}   # save features of all noise levels
    all_noise_features_out = {}  # save features of all noise levels

    for noise_ratio in noise_levels:
        using_noise = False if noise_ratio == 0 else True

        set_seed()

        model, preprocess = clip.load(vit_type + "/" + str(img_patch_size), device=device)
        model.float()
        image_encoder = model.visual

        hidden_dim = image_encoder.ln_post.weight.shape[0]  
        decoder_embed_dim = 512        
        num_decoder_layers = 8        
        num_decoder_heads = 8            
        dropout = 0.1
        decoder = TransformerDecoder(
            encoder_embed_dim=hidden_dim,
            decoder_embed_dim=decoder_embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            dropout=dropout,
            img_size=224,
            patch_size=img_patch_size,
            num_channels=3
        ).to(device)

        image_paths = []
        labels = []
        for idx, id in enumerate(ids):
            for i in range(1):
                image_paths.append(f"../imagenet/tiny-imagenet-200/train/{id}/images/{id}_{i}.JPEG")
                labels.append(idx)

        train_dataset = CustomDataset(image_paths, labels=labels, transform=preprocess)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = CustomDataset(image_paths*5, labels=labels*5, transform=preprocess, noise=True, noise_ratio=noise_ratio)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # create a dictionary to save features of all epochs
        all_epochs_features_in = {}  # save features of all epochs
        all_epochs_features_out = {}  # save features of all epochs
        
        # epoch 0
        print(f"getting data at epoch 0 for noise level {noise_ratio}")
        features_in_dict, features_out_dict = get_data_at_level_and_epoch(image_encoder, val_loader, device, using_noise)
        
        # initialize dictionary structure, create a list to store features of different epochs for each layer
        for layer_name in features_in_dict:
            all_epochs_features_in[layer_name] = [features_in_dict[layer_name]]  # features of epoch 0
            all_epochs_features_out[layer_name] = [features_out_dict[layer_name]]  # features of epoch 0

        for i in range(1, 11):
            epoch = i * eval_epoch
            print(f"getting data at epoch {epoch} for noise level {noise_ratio}")
            ckpt_path = os.path.join(os.path.join(dir, 'ckpt'), f'epoch_{epoch}.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
            
            features_in_dict, features_out_dict = get_data_at_level_and_epoch(image_encoder, val_loader, device, using_noise)
            
            # add current epoch's features to the corresponding list
            for layer_name in features_in_dict:
                all_epochs_features_in[layer_name].append(features_in_dict[layer_name])
                all_epochs_features_out[layer_name].append(features_out_dict[layer_name])
        
        # convert lists to tensors, shape (n_epoch, n_label, n_img, d)
        final_features_in = {}
        final_features_out = {}
        
        for layer_name in all_epochs_features_in:
            # stack all epochs' features: (n_epoch, n_label, n_img, d)
            final_features_in[layer_name] = torch.stack(all_epochs_features_in[layer_name], dim=0)
            final_features_out[layer_name] = torch.stack(all_epochs_features_out[layer_name], dim=0)
            
            print(f"{layer_name} - final input feature shape: {final_features_in[layer_name].shape}")
            print(f"{layer_name} - final output feature shape: {final_features_out[layer_name].shape}")
        
        # add current noise level's features to the dictionary
        all_noise_features_in[noise_ratio] = final_features_in
        all_noise_features_out[noise_ratio] = final_features_out

    # organize all noise level's features to the final format: (n_epoch, n_label, n_noise_ratio, n_img, d)
    final_all_features_in = {}
    final_all_features_out = {}
    
    # get the data of the first noise level to determine the layer names and dimensions
    first_noise_ratio = noise_levels[0]
    layer_names = list(all_noise_features_in[first_noise_ratio].keys())
    
    for layer_name in layer_names:
        # collect features of all noise levels for this layer
        layer_features_in = []
        layer_features_out = []
        
        for noise_ratio in noise_levels:
            # features of current noise level: (n_epoch, n_label, n_img, d)
            layer_features_in.append(all_noise_features_in[noise_ratio][layer_name])
            layer_features_out.append(all_noise_features_out[noise_ratio][layer_name])
        
        # stack features of different noise levels: (n_noise_ratio, n_epoch, n_label, n_img, d)
        stacked_in = torch.stack(layer_features_in, dim=0)
        stacked_out = torch.stack(layer_features_out, dim=0)
        
        # adjust dimension order to: (n_epoch, n_label, n_noise_ratio, n_img, d)
        final_all_features_in[layer_name] = stacked_in.permute(1, 2, 0, 3, 4).numpy()
        final_all_features_out[layer_name] = stacked_out.permute(1, 2, 0, 3, 4).numpy()
        
        print(f"{layer_name} - final input feature shape: {final_all_features_in[layer_name].shape}")
        print(f"{layer_name} - final output feature shape: {final_all_features_out[layer_name].shape}")

    # save the final numpy data
    output_dir = os.path.join(dir, f'manifold_data_{eval_epoch}')
    os.makedirs(output_dir, exist_ok=True)
    
    # save npy files for each layer separately
    for layer_name in final_all_features_in:
        # save input features
        save_path_in = os.path.join(output_dir, f'{layer_name}_features_input.npy')
        np.save(save_path_in, final_all_features_in[layer_name])
        
        # save output features
        save_path_out = os.path.join(output_dir, f'{layer_name}_features_output.npy')
        np.save(save_path_out, final_all_features_out[layer_name])
        
        print(f"{layer_name} features have been saved to:")
        print(f"   input: {save_path_in}")
        print(f"   output: {save_path_out}")
    
    print(f"all layer's features data have been saved to {output_dir}")
    print(f"data format: (n_epoch={final_all_features_in[layer_names[0]].shape[0]}, "
          f"n_label={final_all_features_in[layer_names[0]].shape[1]}, "
          f"n_noise_ratio={final_all_features_in[layer_names[0]].shape[2]}, "
          f"n_img={final_all_features_in[layer_names[0]].shape[3]}, "
          f"d={final_all_features_in[layer_names[0]].shape[4]})")
    print("all noise levels processed!")


if __name__ == "__main__":
    main()



