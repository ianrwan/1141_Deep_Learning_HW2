"""
KSDD2 Surface Defect Detection and Segmentation
Simplified implementation for deep learning homework
"""

import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    # Paths
    try:
        data_root = "./KSDD2"
        train_image_dir = os.path.join(data_root, "train/image")
        train_mask_dir = os.path.join(data_root, "train/ground_truth")
        train_label_dir = os.path.join(data_root, "train/label")
        test_image_dir = os.path.join(data_root, "test/image")
        test_mask_dir = os.path.join(data_root, "test/ground_truth")
        test_label_dir = os.path.join(data_root, "test/label")
    except(FileNotFoundError):
        print("Please check if you download KSDD2 dataset folder.")
    
    # Image parameters
    img_height = 230
    img_width = 630
    resize_height = 256
    resize_width = 640
    
    # Training parameters
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = 10
    num_classes = 2
    
    # Model save paths
    detection_model_path = "detection_model.pth"
    segmentation_model_path = "segmentation_model.pth"

# Custom Dataset for KSDD2
class KSDD2Dataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, label_paths=None, mode='segmentation'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.label_paths = label_paths
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (Config.resize_width, Config.resize_height))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        if self.mode == 'segmentation':
            # Load mask for segmentation
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (Config.resize_width, Config.resize_height))
            mask = (mask > 0).astype(np.float32)  # Convert to binary mask
            mask = np.expand_dims(mask, axis=0)  # Add channel dimension
            
            return torch.FloatTensor(image), torch.FloatTensor(mask)
            
        elif self.mode == 'detection':
            # Load labels for detection
            label_path = self.label_paths[idx]
            boxes = []
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class_id x_center y_center width height
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        # Convert to pixel coordinates
                        x_center *= Config.img_width
                        y_center *= Config.img_height
                        width *= Config.img_width
                        height *= Config.img_height
                        
                        # Convert to (x_min, y_min, x_max, y_max)
                        x_min = x_center - width/2
                        y_min = y_center - height/2
                        x_max = x_center + width/2
                        y_max = y_center + height/2
                        
                        boxes.append([x_min, y_min, x_max, y_max, class_id])
            
            return torch.FloatTensor(image), boxes
    
    @staticmethod
    def collate_fn_detection(batch):
        images, boxes_list = zip(*batch)
        return torch.stack(images), boxes_list

# Simple U-Net model for segmentation
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        output = self.out(dec1)
        return torch.sigmoid(output)

# Data preparation functions
def prepare_segmentation_data():
    """Prepare data for segmentation task"""
    train_image_paths = sorted(glob.glob(os.path.join(Config.train_image_dir, "*.png")))
    train_mask_paths = sorted(glob.glob(os.path.join(Config.train_mask_dir, "*.png")))
    test_image_paths = sorted(glob.glob(os.path.join(Config.test_image_dir, "*.png")))
    test_mask_paths = sorted(glob.glob(os.path.join(Config.test_mask_dir, "*.png")))
    
    # Filter to ensure matching pairs
    train_pairs = []
    test_pairs = []
    
    for img_path in train_image_paths:
        base_name = os.path.basename(img_path).replace(".png", "")
        mask_path = os.path.join(Config.train_mask_dir, f"{base_name}_GT.png")
        if os.path.exists(mask_path):
            train_pairs.append((img_path, mask_path))
    
    for img_path in test_image_paths:
        base_name = os.path.basename(img_path).replace(".png", "")
        mask_path = os.path.join(Config.test_mask_dir, f"{base_name}_GT.png")
        if os.path.exists(mask_path):
            test_pairs.append((img_path, mask_path))
    
    train_images, train_masks = zip(*train_pairs)
    test_images, test_masks = zip(*test_pairs)
    
    return list(train_images), list(train_masks), list(test_images), list(test_masks)

def prepare_detection_data():
    """Prepare data for detection task"""
    train_image_paths = sorted(glob.glob(os.path.join(Config.train_image_dir, "*.png")))
    test_image_paths = sorted(glob.glob(os.path.join(Config.test_image_dir, "*.png")))
    
    # Find corresponding label files
    train_data = []
    test_data = []
    
    for img_path in train_image_paths:
        base_name = os.path.basename(img_path).replace(".png", "")
        label_path = os.path.join(Config.train_label_dir, f"{base_name}.txt")
        train_data.append((img_path, label_path))
    
    for img_path in test_image_paths:
        base_name = os.path.basename(img_path).replace(".png", "")
        label_path = os.path.join(Config.test_label_dir, f"{base_name}.txt")
        test_data.append((img_path, label_path))
    
    train_images, train_labels = zip(*train_data)
    test_images, test_labels = zip(*test_data)
    
    return list(train_images), list(train_labels), list(test_images), list(test_labels)

# Training functions
def train_segmentation_model():
    """Train segmentation model"""
    print("Preparing segmentation data...")
    train_images, train_masks, test_images, test_masks = prepare_segmentation_data()
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create datasets
    train_dataset = KSDD2Dataset(train_images, train_masks, mode='segmentation')
    test_dataset = KSDD2Dataset(test_images, test_masks, mode='segmentation')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # Initialize model
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Training loop
    print("Starting segmentation training...")
    for epoch in range(Config.num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), Config.segmentation_model_path)
    print(f"Segmentation model saved to {Config.segmentation_model_path}")
    
    return model

def train_detection_model():
    """Train detection model"""
    print("Preparing detection data...")
    train_images, train_labels, test_images, test_labels = prepare_detection_data()
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    
    # Create datasets
    train_dataset = KSDD2Dataset(train_images, label_paths=train_labels, mode='detection')
    test_dataset = KSDD2Dataset(test_images, label_paths=test_labels, mode='detection')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, 
                             collate_fn=KSDD2Dataset.collate_fn_detection)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False,
                            collate_fn=KSDD2Dataset.collate_fn_detection)
    
    # Initialize model
    model = UNet().to(device)  # Using same UNet for simplicity
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Training loop
    print("Starting detection training...")
    for epoch in range(min(Config.num_epochs, 10)):  # Fewer epochs for detection
        model.train()
        train_loss = 0.0
        
        for images, boxes_list in tqdm(train_loader, desc=f"Epoch {epoch+1}/{min(Config.num_epochs, 10)}"):
            images = images.to(device)
            
            # Create dummy masks for training (simplified)
            batch_size = images.shape[0]
            dummy_masks = torch.zeros((batch_size, 1, Config.resize_height, Config.resize_width)).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, dummy_masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), Config.detection_model_path)
    print(f"Detection model saved to {Config.detection_model_path}")
    
    return model

# Evaluation functions
def dice_coefficient(pred, target):
    """Calculate Dice coefficient"""
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def evaluate_segmentation(model):
    """Evaluate segmentation model"""
    print("Evaluating segmentation model...")
    
    _, _, test_images, test_masks = prepare_segmentation_data()
    
    test_dataset = KSDD2Dataset(test_images, test_masks, mode='segmentation')
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            for i in range(preds.shape[0]):
                dice = dice_coefficient(preds[i], masks[i])
                dice_scores.append(dice.item())
    
    avg_dice = np.mean(dice_scores)
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    return avg_dice

def visualize_results():
    """Visualize some results without matplotlib if possible"""
    print("Visualizing results...")
    
    # Load a sample image
    test_images, test_masks = prepare_segmentation_data()[2:4]
    
    if len(test_images) > 0:
        # Load and display first sample using OpenCV instead of matplotlib
        image = cv2.imread(test_images[0])
        mask = cv2.imread(test_masks[0], cv2.IMREAD_GRAYSCALE)
        
        # Create overlay
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 0] = 0  # Set blue channel to 0
        mask_color[:, :, 1] = 255  # Set green channel to max (green mask)
        mask_color[:, :, 2] = 0  # Set red channel to 0
        
        # Overlay mask on image
        overlay = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
        
        # Save results
        cv2.imwrite("original_image.png", image)
        cv2.imwrite("ground_truth_mask.png", mask)
        cv2.imwrite("overlay_result.png", overlay)
        
        print("Results saved as:")
        print("  - original_image.png")
        print("  - ground_truth_mask.png")
        print("  - overlay_result.png")
        
        # Option 1: Simple display using OpenCV (commented out to avoid window issues)
        # cv2.imshow("Original Image", image)
        # cv2.imshow("Ground Truth Mask", mask)
        # cv2.imshow("Overlay", overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Option 2: Print image info without displaying
        print(f"\nSample image info:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask unique values: {np.unique(mask)}")
        
        # Count defect pixels
        defect_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        defect_ratio = defect_pixels / total_pixels * 100
        print(f"  Defect pixels: {defect_pixels} ({defect_ratio:.2f}% of image)")
        
        return True
    else:
        print("No test images found!")
        return False

# Main execution
def main():
    print("=" * 50)
    print("KSDD2 Surface Defect Detection and Segmentation")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists(Config.data_root):
        print(f"Error: Data directory not found at {Config.data_root}")
        print("Please ensure KSDD2 dataset is downloaded and placed in the correct location.")
        return
    
    try:
        # Train or load models
        train_new_models = True
        
        if train_new_models:
            print("\n1. Training Segmentation Model (U-Net)...")
            seg_model = train_segmentation_model()
            
            print("\n2. Training Detection Model (Simplified)...")
            det_model = train_detection_model()
        else:
            # Load existing models
            seg_model = UNet().to(device)
            seg_model.load_state_dict(torch.load(Config.segmentation_model_path, map_location=device))
            seg_model.eval()
            
            det_model = UNet().to(device)
            det_model.load_state_dict(torch.load(Config.detection_model_path, map_location=device))
            det_model.eval()
        
        # Evaluate models
        print("\n3. Evaluating Models...")
        dice_score = evaluate_segmentation(seg_model)
        
        # Note: Detection evaluation
        print("\nNote: Detection model is simplified for demonstration.")
        print("For proper detection, use YOLO or Faster R-CNN.")
        
        # Visualize
        print("\n4. Visualizing Sample Results...")
        visualize_results()
        
        print("\n" + "=" * 50)
        print("Program completed successfully!")
        print(f"Segmentation Dice Coefficient: {dice_score:.4f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()