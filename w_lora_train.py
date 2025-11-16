import torch
import clip
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
from torch.optim import AdamW
import torch.nn.functional as F
import os
import random
import numpy as np
from evaluation import *
from dataset import CustomDataset
import torch.nn as nn
from decoder import TransformerDecoder
from utils import *
from lora import *
import json
from config import ids
import argparse


def extract_final_block_features(image_encoder, images):
    """
    Use hook to intercept the output features of the last transformer block in image_encoder,
    the output shape is [seq_len, batch_size, hidden_dim] (not through the last mapping layer).
    """
    container = {}
    last_block = image_encoder.transformer.resblocks[-1]

    def hook_fn(module, inp, outp):
        container["final"] = outp

    handle = last_block.register_forward_hook(hook_fn)
    _ = image_encoder(images)
    handle.remove()
    return container["final"]


def train_one_epoch(image_encoder, decoder, train_loader, optimizer, device):
    image_encoder.train()
    decoder.train()
    total_loss = 0.0
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    for clean_img, _, _ in train_loader:
        clean_img = clean_img.to(device)

        features = extract_final_block_features(image_encoder, clean_img)
        # decoder reconstruct the image
        reconstructed = decoder(features)
        clean_img = unnormalize(clean_img, device)
        
        # calculate the loss in the normalized space
        mse_loss = mse_criterion(reconstructed, clean_img)
        l1_loss = l1_criterion(reconstructed, clean_img)
        loss = mse_loss + 0.1 * l1_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def train(
    image_encoder,
    decoder,
    train_loader,
    device,
    num_epochs=25,
    lr=1e-4,
    log_dir="logs",
):
    """
    Training process:
      - Use train_one_epoch to train encoder+decoder (add noise inside) for each epoch;
      - After each epoch, use the existing train_regression_layer and validate_regression_layer to verify the encoder,
        the verification process remains consistent with the previous one.
    """
    optimizer = AdamW(list(image_encoder.parameters()) + list(decoder.parameters()), lr=lr)
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(image_encoder, decoder, train_loader, optimizer, device)
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        if epoch % (num_epochs // 5) == 0 or epoch == num_epochs: # or (epoch < (num_epochs // 5) and epoch % 2 == 0):
            # Save the model checkpoint
            ckpt_dir = os.path.join(log_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'image_encoder_state_dict': image_encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, ckpt_path)
            
            logging.info(f"Model checkpoint saved to {ckpt_path}")


def main(lora_ranks=[4,8], train_epoch=500, decoder_pretrained=False, decoder_pretrained_path=None, vit_type="ViT-B", img_patch_size=32, batch_size=8, lr=2e-4, device="cuda"):
    set_seed()
    

    image_paths = []
    labels = []
    for idx, id in enumerate(ids):
        for i in range(1):
            image_paths.append(f"../imagenet/tiny-imagenet-200/train/{id}/images/{id}_{i}.JPEG")
            labels.append(idx)

    model, preprocess = clip.load(vit_type + "/" + str(img_patch_size), device=device)

    train_dataset = CustomDataset(image_paths, labels=labels, transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for lrk in lora_ranks:
        model, preprocess = clip.load(vit_type + "/" + str(img_patch_size), device=device)
        model.float()
        image_encoder = model.visual

        lora_alpha=lrk*4      # alpha of lora
        lora_dropout=0.0    # dropout of lora

        replace_linear_with_lora_mha_inproj(image_encoder, lrk, lora_alpha, lora_dropout)

        # train prameters only except the initial Wk, Wq, Wv
        mark_initial_weight_as_not_trainable(image_encoder, "in_proj")

        # Build decoder for image reconstruction
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

        log_dir = f"w_lora"
        log_dir = os.path.join(log_dir, "lora_in_proj_initial_not_train", f"lora_r{lrk}")
        train(image_encoder, decoder, train_loader, device, num_epochs=train_epoch, lr=lr, log_dir=log_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_ranks", type=str2list, default=[8])
    parser.add_argument("--train_epoch", type=int, default=100)
    parser.add_argument("--decoder_pretrained", type=str2bool, default=False)
    parser.add_argument("--decoder_pretrained_path", type=str, default=None)
    parser.add_argument("--vit_type", type=str, default="ViT-B")
    parser.add_argument("--img_patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(lora_ranks=args.lora_ranks, 
         train_epoch=args.train_epoch, 
         decoder_pretrained=args.decoder_pretrained, 
         decoder_pretrained_path=args.decoder_pretrained_path,
         vit_type=args.vit_type,
         img_patch_size=args.img_patch_size,
         batch_size=args.batch_size,
         lr=args.lr,
         device=args.device)
