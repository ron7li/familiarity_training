import torch
import clip
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
from torch.optim import Adam
import torch.nn.functional as F
import os
import random
import numpy as np
from evaluation import *
from dataset import CustomDataset  
import torch.nn as nn
from decoder import TransformerDecoder
from utils import *
from config import ids
import json
import argparse
import matplotlib.pyplot as plt


def validate(image_encoder, 
             decoder, 
             train_loader, 
             val_loader, 
             device, 
             eval_epoch, 
             log_dir, 
             using_noise=True, 
             linear_lr=1e-4, 
             linear_training_epochs=100,
             if_attention_map=False):
    layer_acc_outputs = []
    layer_prob_outputs = []
    layer_cos_outputs = []
    layer_pcc_outputs = []
    cpts = []

    # epoch 0
    logging.info("== Validation of epoch 0: building layer classification... ==")
    image_encoder.eval()
    layer2classification = train_classification_layer(
        image_encoder=image_encoder,
        train_loader=train_loader,
        device=device,
        epochs=linear_training_epochs,
        lr=linear_lr
    )
    layer_acc, layer_prob = validate_classification_layer(
        image_encoder=image_encoder,
        val_loader=val_loader,
        layer2classification=layer2classification,
        device=device,
    )

    feature_save_dir = os.path.join(log_dir, f'feature_characteristics/epoch_0')
    os.makedirs(feature_save_dir, exist_ok=True)

    result = validate_layer_features(
        image_encoder=image_encoder,
        val_loader=val_loader,
        device=device,
        visualize=False,
        save_dir=feature_save_dir,
        using_noise=using_noise,
        validate_correlation=False,
        use_attention_maps=if_attention_map,
        if_save_attention_maps=True
    )
    layer_cos, layer_pcc = result[:2]
    layer_acc_outputs.append(layer_acc)
    layer_prob_outputs.append(layer_prob)
    layer_cos_outputs.append(layer_cos)
    layer_pcc_outputs.append(layer_pcc)
    cpts.append(0)

    for i in range(1, 6):
        epoch = eval_epoch * i
        logging.info(f"== Validation of epoch {epoch}: building layer classification... ==")
        if_save_attention_maps = True if i == 5 else False
        
        # load model weights
        ckpt_path = os.path.join(os.path.join(log_dir.split('/')[0], 'ckpt'), f'epoch_{epoch}.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        image_encoder.eval()
        layer2classification = train_classification_layer(
            image_encoder=image_encoder,
            train_loader=train_loader, 
            device=device,
            epochs=linear_training_epochs,
            lr=linear_lr
        )
        layer_acc, layer_prob = validate_classification_layer(
            image_encoder=image_encoder,
            val_loader=val_loader,
            layer2classification=layer2classification,
            device=device,
        )

        feature_save_dir = os.path.join(log_dir, f'feature_characteristics/epoch_{epoch}')
        os.makedirs(feature_save_dir, exist_ok=True)

        result = validate_layer_features(
            image_encoder=image_encoder,
            val_loader=val_loader,
            device=device,
            visualize=False,
            save_dir=feature_save_dir,
            using_noise=using_noise,
            validate_correlation=False,
            use_attention_maps=if_attention_map,
            if_save_attention_maps=if_save_attention_maps
        )
        layer_cos, layer_pcc = result[:2]
        layer_acc_outputs.append(layer_acc)
        layer_prob_outputs.append(layer_prob)
        layer_cos_outputs.append(layer_cos)
        layer_pcc_outputs.append(layer_pcc)
        cpts.append(epoch)

    # save validation results
    acc_save_path = os.path.join(log_dir, "layer_acc_outputs.png")
    prob_save_path = os.path.join(log_dir, "layer_prob_outputs.png")
    cos_save_path = os.path.join(log_dir, "layer_cos_outputs.png")
    pcc_save_path = os.path.join(log_dir, "layer_pcc_outputs.png")
    plot_dicts_subplots(layer_acc_outputs, cpts, acc_save_path, mode="Accuracy")
    logging.info(f"ACC Plot saved to {acc_save_path}")
    plot_dicts_subplots(layer_prob_outputs, cpts, prob_save_path, mode="Correct Class Prediction Probability")
    logging.info(f"Prob Plot saved to {prob_save_path}")
    plot_dicts_subplots(layer_cos_outputs, cpts, cos_save_path, mode="Cosine Similarity")
    logging.info(f"Cos Plot saved to {cos_save_path}")
    plot_dicts_subplots(layer_pcc_outputs, cpts, pcc_save_path, mode="Pearson Correlation Coefficient")
    logging.info(f"PCC Plot saved to {pcc_save_path}")
    
    return layer_acc_outputs, layer_prob_outputs, layer_cos_outputs, layer_pcc_outputs


# show reconstruction results
def show_reconstruction(image_encoder, decoder, val_loader, device, log_dir, n_samples=4):
    """
    show the comparison between original images and reconstructed images
    """
    image_encoder.eval()
    decoder.eval()
    
    # create save directory
    save_dir = os.path.join(log_dir, 'reconstruction_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # get n_samples samples
    images_a = []
    images_b = []
    recon_images = []
    
    # list for calculating losses
    mse_losses = []
    l1_losses = []
    
    # define loss functions
    mse_criterion = torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()
    
    with torch.no_grad():
        for img_a, img_b, _ in val_loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            
            # get encoded features
            features = extract_final_block_features(image_encoder, img_b)
            # reconstruct
            recon = decoder(features)
            
            # calculate reconstruction losses (reconstructed images vs clean images img_a)
            mse_loss = mse_criterion(recon[:n_samples], img_a[:n_samples]).item()
            l1_loss = l1_criterion(recon[:n_samples], img_a[:n_samples]).item()
            
            mse_losses.append(mse_loss)
            l1_losses.append(l1_loss)
            
            # convert to numpy arrays for visualization
            img_a_np = unnormalize(img_a, device).clamp(0, 1).cpu().numpy()
            img_b_np = unnormalize(img_b, device).clamp(0, 1).cpu().numpy()
            recon_np = recon.clamp(0, 1).cpu().numpy()
            
            # add current batch images to lists
            for j in range(img_a_np.shape[0]):
                if len(images_a) >= n_samples:
                    break
                images_a.append(img_a_np[j:j+1])
                images_b.append(img_b_np[j:j+1])
                recon_images.append(recon_np[j:j+1])
            
            if len(images_a) >= n_samples:
                break
    
    # calculate average losses
    avg_mse_loss = sum(mse_losses) / len(mse_losses)
    avg_l1_loss = sum(l1_losses) / len(l1_losses)
    
    # print loss results
    print(f"reconstruction losses (reconstructed images vs clean images):")
    print(f"   average MSE loss: {avg_mse_loss:.6f}")
    print(f"   average L1 loss:  {avg_l1_loss:.6f}")
    print(f"   number of samples:    {len(mse_losses)}")
    
    # plot
    plt.figure(figsize=(15, 3*n_samples))
    for i in range(n_samples):
        # original image
        plt.subplot(n_samples, 3, i*3 + 1)
        plt.imshow(np.transpose(images_a[i][0], (1,2,0)))
        plt.title('original_image')
        plt.axis('off')
        
        # noisy image
        plt.subplot(n_samples, 3, i*3 + 2)
        plt.imshow(np.transpose(images_b[i][0], (1,2,0)))
        plt.title('noisy_image')
        plt.axis('off')
        
        # reconstructed image
        plt.subplot(n_samples, 3, i*3 + 3)
        plt.imshow(np.transpose(recon_images[i][0], (1,2,0)))
        plt.title('reconstructed_image')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_results.png'))
    plt.close()
    
    # save loss statistics to file
    with open(os.path.join(save_dir, 'reconstruction_losses.txt'), 'w') as f:
        f.write(f"reconstruction losses (reconstructed images vs clean images):\n")
        f.write(f"average MSE loss: {avg_mse_loss:.6f}\n")
        f.write(f"average L1 loss:  {avg_l1_loss:.6f}\n")
        f.write(f"number of samples:    {len(mse_losses)}\n")
        f.write(f"\ndetailed losses (each batch):\n")
        for i, (mse, l1) in enumerate(zip(mse_losses, l1_losses)):
            f.write(f"Batch {i+1}: MSE={mse:.6f}, L1={l1:.6f}\n")
    
    print(f"reconstruction results saved to: {save_dir}")
    print(f"loss statistics saved to: {os.path.join(save_dir, 'reconstruction_losses.txt')}")
    
    return avg_mse_loss, avg_l1_loss


def main(using_noise=True, 
         noise_ratio=0.3, 
         if_show_reconstruction=False, 
         eval_epoch=100, 
         decoder_pretrained=False, 
         decoder_pretrained_path=None, 
         vit_type="ViT-B", 
         img_patch_size=16,
         batch_size=8,
         linear_lr=1e-4,
         linear_training_epochs=100,
         num_noise_pattern=5,
         if_attention_map=False,
         device="cuda"):
    set_seed()
    

    # load CLIP model (take visual encoder part)
    model, preprocess = clip.load(vit_type + "/" + str(img_patch_size), device=device)
    model.float()
    image_encoder = model.visual
    
    # build transformer decoder, set parameters refer to MAE decoder structure
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

    if decoder_pretrained and decoder_pretrained_path is not None:
        logging.info(f"Loading pretrained decoder from {decoder_pretrained_path}")
        checkpoint = torch.load(decoder_pretrained_path, map_location=device)
        if 'decoder_state_dict' in checkpoint:
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            decoder.load_state_dict(checkpoint)
    
    # construct dataset (here only load clean images, no noise processing)
    image_paths = []
    labels = []
    for idx, id in enumerate(ids):
        for i in range(1):
            image_paths.append(f"../imagenet/tiny-imagenet-200/train/{id}/images/{id}_{i}.JPEG")
            labels.append(idx)

    train_dataset = CustomDataset(image_paths, labels=labels, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if using_noise:
        val_dataset = CustomDataset(image_paths*num_noise_pattern, labels=labels*num_noise_pattern, transform=preprocess, noise=True, noise_ratio=noise_ratio)
    else:
        val_dataset = CustomDataset(image_paths, labels=labels, transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # log directory
    if using_noise:
        if num_noise_pattern == 5:
            log_dir = f"wo_lora/noise_ratio_{noise_ratio}/eval_epoch_{eval_epoch}"
        else:
            log_dir = f"wo_lora/noise_ratio_{noise_ratio}/eval_epoch_{eval_epoch}_num_noise_pattern_{num_noise_pattern}"
    else:
        log_dir = f"wo_lora/clean/eval_epoch_{eval_epoch}"
    os.makedirs(log_dir, exist_ok=True)

    layer_acc_outputs, layer_prob_outputs, layer_cos_outputs, layer_pcc_outputs = validate(
        image_encoder=image_encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        eval_epoch=eval_epoch,
        log_dir=log_dir,
        using_noise=using_noise,
        linear_lr=linear_lr,
        linear_training_epochs=linear_training_epochs,
        if_attention_map=if_attention_map
    )

    if if_show_reconstruction:
        show_reconstruction(image_encoder, decoder, val_loader, device, log_dir, n_samples=4)

    output_dic = {"wo_lora": (layer_acc_outputs, layer_prob_outputs, layer_cos_outputs, layer_pcc_outputs)}
    # save json
    data_serializable = {}
    for k, (acc_list, prob_list, cos_list, pcc_list) in output_dic.items():
        acc_serial = []
        for d in acc_list:
            acc_serial.append({kk: float(vv) for kk, vv in d.items()})

        prob_serial = []
        for d in prob_list:
            prob_serial.append({kk: float(vv) for kk, vv in d.items()})
            
        cos_serial = []
        for d in cos_list:
            cos_serial.append({kk: float(vv) for kk, vv in d.items()})
            
        pcc_serial = []
        for d in pcc_list:
            pcc_serial.append({kk: float(vv) for kk, vv in d.items()})

        data_serializable[k] = {
            "acc": acc_serial,
            "prob": prob_serial,
            "cos": cos_serial,
            "pcc": pcc_serial
        }

    file_dir = os.path.join(log_dir, "output_data")
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, "lora_results.json")

    with open(file_path, "w") as f:
        json.dump(data_serializable, f, indent=2)

    print(f"data saved to {file_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--using_noise", type=str2bool, default=True)
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--if_show_reconstruction", type=str2bool, default=True)
    parser.add_argument("--eval_epoch", type=int, default=20)
    parser.add_argument("--decoder_pretrained", type=str2bool, default=False)
    parser.add_argument("--decoder_pretrained_path", type=str, default=None)
    parser.add_argument("--vit_type", type=str, default="ViT-B")
    parser.add_argument("--img_patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--linear_lr", type=float, default=1e-4)
    parser.add_argument("--linear_training_epochs", type=int, default=100)
    parser.add_argument("--num_noise_pattern", type=int, default=5)
    parser.add_argument("--if_attention_map", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    print(args.using_noise)
    main(using_noise=args.using_noise, 
         noise_ratio=args.noise_ratio, 
         if_show_reconstruction=args.if_show_reconstruction, 
         eval_epoch=args.eval_epoch, 
         decoder_pretrained=args.decoder_pretrained, 
         decoder_pretrained_path=args.decoder_pretrained_path,
         vit_type=args.vit_type, 
         img_patch_size=args.img_patch_size,
         batch_size=args.batch_size,
         linear_lr=args.linear_lr,
         linear_training_epochs=args.linear_training_epochs,
         num_noise_pattern=args.num_noise_pattern,
         if_attention_map=args.if_attention_map,
         device=args.device)