import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import HDRPointwiseNN

class RealEstateDataset(Dataset):
    """Dataset for real estate photo enhancement using _src/_tar pairs"""
    
    def __init__(self, data_dir, split='train', image_size=256, validation_split=0.1):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Find all src/tar pairs
        src_files = sorted(glob.glob(os.path.join(data_dir, '*_src.jpg')))
        tar_files = sorted(glob.glob(os.path.join(data_dir, '*_tar.jpg')))
        
        # Match pairs by removing _src/_tar and comparing base names
        pairs = []
        for src_file in src_files:
            base_name = src_file.replace('_src.jpg', '')
            tar_file = base_name + '_tar.jpg'
            if tar_file in tar_files:
                pairs.append((src_file, tar_file))
        
        # Split into train/validation
        split_idx = int(len(pairs) * (1 - validation_split))
        if split == 'train':
            self.pairs = pairs[:split_idx]
        else:
            self.pairs = pairs[split_idx:]
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f"Found {len(self.pairs)} real estate photo pairs for {split}")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_path, tar_path = self.pairs[idx]
        
        try:
            # Load source (original) and target (enhanced) images
            src_img = Image.open(src_path).convert('RGB')
            tar_img = Image.open(tar_path).convert('RGB')
            
            # Convert to tensors
            src_tensor = self.transform(src_img)
            tar_tensor = self.transform(tar_img)
            
            # Create low-res version for HDRNet
            lowres_tensor = transforms.functional.resize(src_tensor, (64, 64))
            
            return {
                'lowres': lowres_tensor,      # 64x64 for bilateral grid
                'fullres': src_tensor,        # 256x256 source image
                'target': tar_tensor          # 256x256 enhanced target
            }
            
        except Exception as e:
            print(f"Error loading {src_path}: {e}")
            # Return dummy data on error
            dummy = torch.zeros(3, self.image_size, self.image_size)
            return {
                'lowres': transforms.functional.resize(dummy, (64, 64)),
                'fullres': dummy,
                'target': dummy
            }

def save_comparison_images(model, dataloader, device, save_path, epoch):
    """Save before/after comparison images"""
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        lowres = batch['lowres'][:4].to(device)
        fullres = batch['fullres'][:4].to(device)
        target = batch['target'][:4].to(device)
        
        # Generate enhanced images
        enhanced = model(lowres, fullres)
        
        # Create comparison plot
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i in range(4):
            # Original
            axes[0, i].imshow(fullres[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Target (professional edit)
            axes[1, i].imshow(target[i].cpu().permute(1, 2, 0))
            axes[1, i].set_title('Target (Pro Edit)')
            axes[1, i].axis('off')
            
            # AI Enhanced
            axes[2, i].imshow(torch.clamp(enhanced[i].cpu().permute(1, 2, 0), 0, 1))
            axes[2, i].set_title('AI Enhanced')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def train_real_estate_enhancer():
    """Train HDRNet on real estate photos"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Model parameters optimized for real estate photos
    params = {
        'guide_complexity': 16,
        'luma_bins': 8,
        'channel_multiplier': 1,
        'spatial_bin': 16,
        'batch_norm': True,
        'net_input_size': 64
    }
    
    # Create model and training components
    model = HDRPointwiseNN(params).to(device)
    criterion = nn.L1Loss()  # L1 loss works well for photo enhancement
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = RealEstateDataset('photos_to_process', 'train', image_size=256)
    val_dataset = RealEstateDataset('photos_to_process', 'val', image_size=256)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Training on {len(train_dataset)} pairs, validating on {len(val_dataset)} pairs")
    
    # Create save directories
    os.makedirs('real_estate_checkpoints', exist_ok=True)
    os.makedirs('real_estate_checkpoints/samples', exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    epochs = 50
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc='Training')
        
        for batch in train_pbar:
            lowres = batch['lowres'].to(device)
            fullres = batch['fullres'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(lowres, fullres)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                lowres = batch['lowres'].to(device)
                fullres = batch['fullres'].to(device)
                target = batch['target'].to(device)
                
                output = model(lowres, fullres)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'real_estate_checkpoints/best_real_estate_enhancer.pth')
            print("💾 Saved new best model!")
        
        # Save sample images every 10 epochs
        if epoch % 10 == 0:
            sample_path = f'real_estate_checkpoints/samples/epoch_{epoch:03d}.png'
            save_comparison_images(model, val_loader, device, sample_path, epoch)
            print(f"📸 Saved sample comparisons to {sample_path}")
    
    print("\n🎉 Real estate enhancement training completed!")
    print(f"📁 Best model saved to: real_estate_checkpoints/best_real_estate_enhancer.pth")
    print(f"📊 Sample images saved to: real_estate_checkpoints/samples/")

if __name__ == '__main__':
    train_real_estate_enhancer()
