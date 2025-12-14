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

class HDRDataset(Dataset):
    """Dataset for HDR image enhancement training"""
    
    def __init__(self, data_dir, split='train', image_size=256, augment=True):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Find input/output image pairs
        self.image_pairs = self._find_image_pairs()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        print(f"Found {len(self.image_pairs)} image pairs for {split}")
        
    def _find_image_pairs(self):
        """Find input/output image pairs in the dataset directory"""
        pairs = []
        
        input_dir = os.path.join(self.data_dir, self.split, 'input')
        output_dir = os.path.join(self.data_dir, self.split, 'output')
        
        if os.path.exists(input_dir) and os.path.exists(output_dir):
            input_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                               glob.glob(os.path.join(input_dir, '*.png')))
            
            for input_path in input_files:
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, filename)
                if os.path.exists(output_path):
                    pairs.append((input_path, output_path))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        input_path, output_path = self.image_pairs[idx]
        
        try:
            # Load images
            input_img = Image.open(input_path).convert('RGB')
            output_img = Image.open(output_path).convert('RGB')
            
            # Convert to tensors
            input_tensor = self.transform(input_img)
            output_tensor = self.transform(output_img)
            
            # Create low-res version for HDRNet (64x64)
            lowres_tensor = transforms.functional.resize(input_tensor, (64, 64))
            
            return {
                'lowres': lowres_tensor,
                'fullres': input_tensor,
                'target': output_tensor
            }
            
        except Exception as e:
            print(f"Error loading {input_path}: {e}")
            # Return dummy data on error
            dummy = torch.zeros(3, self.image_size, self.image_size)
            return {
                'lowres': transforms.functional.resize(dummy, (64, 64)),
                'fullres': dummy,
                'target': dummy
            }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        lowres = batch['lowres'].to(device)
        fullres = batch['fullres'].to(device)
        target = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(lowres, fullres)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            lowres = batch['lowres'].to(device)
            fullres = batch['fullres'].to(device)
            target = batch['target'].to(device)
            
            output = model(lowres, fullres)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_sample_images(model, dataloader, device, save_dir, epoch):
    """Save sample enhanced images"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        lowres = batch['lowres'][:4].to(device)
        fullres = batch['fullres'][:4].to(device)
        target = batch['target'][:4].to(device)
        
        output = model(lowres, fullres)
        
        # Save comparison
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i in range(4):
            # Input
            axes[0, i].imshow(fullres[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title('Input')
            axes[0, i].axis('off')
            
            # Target
            axes[1, i].imshow(target[i].cpu().permute(1, 2, 0))
            axes[1, i].set_title('Target')
            axes[1, i].axis('off')
            
            # Output
            axes[2, i].imshow(torch.clamp(output[i].cpu().permute(1, 2, 0), 0, 1))
            axes[2, i].set_title('Enhanced')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'), dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train HDRNet')
    parser.add_argument('--data-dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=256, help='Image size')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    params = {
        'guide_complexity': 16,
        'luma_bins': 8,
        'channel_multiplier': 1,
        'spatial_bin': 16,
        'batch_norm': True,
        'net_input_size': 64
    }
    
    # Create model
    model = HDRPointwiseNN(params).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = HDRDataset(args.data_dir, 'train', args.image_size)
    val_dataset = HDRDataset(args.data_dir, 'val', args.image_size, augment=False)
    
    if len(train_dataset) == 0:
        print("❌ No training data found! Please check your dataset structure.")
        print("Expected structure:")
        print("data/")
        print("├── train/")
        print("│   ├── input/")
        print("│   └── output/") 
        print("└── val/")
        print("    ├── input/")
        print("    └── output/")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if len(val_loader) > 0:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print("💾 Saved best model!")
        
        # Save samples every 5 epochs
        if epoch % 5 == 0:
            loader = val_loader if len(val_loader) > 0 else train_loader
            save_sample_images(model, loader, device, os.path.join(args.save_dir, 'samples'), epoch)
    
    print("\n🎉 Training completed!")

if __name__ == '__main__':
    main()
