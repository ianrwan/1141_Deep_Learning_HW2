"""
detect.py - Object Detection for KSDD2 dataset
Improved version with mAP and comprehensive evaluation metrics
"""

import os
# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import json
from collections import defaultdict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    # Paths
    try:
        data_root = "./KSDD2"
        test_image_dir = os.path.join(data_root, "test/image")
        test_label_dir = os.path.join(data_root, "test/label")
    except(FileNotFoundError):
        print("Please check if you download KSDD2 dataset folder.")
    
    # Image parameters
    img_height = 230
    img_width = 630
    
    # Model paths
    segmentation_model_path = "segmentation_model.pth"
    output_dir = "detection_results"
    
    # Detection parameters
    confidence_threshold = 0.1
    min_area = 10
    
    # Evaluation parameters
    iou_thresholds = [0.5]  # IoU thresholds for mAP calculation

# U-Net model (same as in main.py)
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

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_dice_coefficient(pred_mask, gt_mask):
    """Calculate Dice coefficient for segmentation"""
    smooth = 1e-6
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def calculate_pixel_accuracy(pred_mask, gt_mask):
    """Calculate pixel-level accuracy"""
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = pred_mask.size
    accuracy = correct_pixels / total_pixels
    return accuracy

def calculate_precision_recall_f1(pred_mask, gt_mask):
    """Calculate precision, recall, and F1-score for segmentation"""
    smooth = 1e-6
    
    # True Positives, False Positives, False Negatives
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1_score = 2 * precision * recall / (precision + recall + smooth)
    
    return precision, recall, f1_score

def evaluate_detection(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Evaluate object detection performance for single image"""
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0  # Perfect score if both empty
    
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0, 0, 0, len(gt_boxes)  # No predictions
    
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0, 0, len(pred_boxes), 0  # All predictions are false positives
    
    # Match predictions to ground truth
    matched_gt = set()
    matched_pred = set()
    
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
                
            iou = calculate_iou(pred_box[:4], gt_box[:4])
            if iou >= iou_threshold:
                matched_gt.add(j)
                matched_pred.add(i)
                break
    
    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score, tp, fp, fn

def calculate_average_precision(recalls, precisions):
    """Calculate Average Precision using the precision-recall curve"""
    # Ensure arrays are sorted by recall
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    # Add sentinel values at both ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under PR curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    
    return ap

def calculate_map(predictions_dict, ground_truths_dict, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        predictions_dict: dict with image_id as key and list of predictions as value
                         each prediction is [x1, y1, x2, y2, confidence]
        ground_truths_dict: dict with image_id as key and list of ground truth boxes as value
                           each ground truth is [x1, y1, x2, y2]
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        mAP value
    """
    # Collect all predictions across all images
    all_predictions = []
    all_ground_truths = []
    
    for image_id in predictions_dict:
        preds = predictions_dict[image_id]
        gts = ground_truths_dict.get(image_id, [])
        
        # Add image_id to predictions
        for pred in preds:
            if len(pred) >= 5:  # Ensure we have confidence score
                all_predictions.append({
                    'image_id': image_id,
                    'bbox': pred[:4],
                    'confidence': pred[4],
                    'used': False
                })
        
        # Add image_id to ground truths
        for gt in gts:
            if len(gt) >= 4:  # Ensure we have valid bbox
                all_ground_truths.append({
                    'image_id': image_id,
                    'bbox': gt[:4],
                    'matched': False
                })
    
    # Sort predictions by confidence (descending)
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calculate precision-recall curve
    tp = np.zeros(len(all_predictions))
    fp = np.zeros(len(all_predictions))
    
    # Match predictions to ground truths
    for i, pred in enumerate(all_predictions):
        image_id = pred['image_id']
        pred_bbox = pred['bbox']
        
        # Find ground truths for this image
        image_gts = [gt for gt in all_ground_truths if gt['image_id'] == image_id]
        
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(image_gts):
            if gt['matched']:
                continue
                
            iou = calculate_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        # Check if the prediction is correct
        if best_iou >= iou_threshold:
            tp[i] = 1
            image_gts[best_gt_idx]['matched'] = True
        else:
            fp[i] = 1
    
    # Calculate cumulative precision and recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls = cum_tp / (len(all_ground_truths) + 1e-6)
    
    # Calculate Average Precision
    ap = calculate_average_precision(recalls, precisions)
    
    return ap

def analyze_model_output():
    """Analyze what the model is actually outputting with comprehensive metrics"""
    print("\n" + "=" * 60)
    print("MODEL OUTPUT ANALYSIS WITH COMPREHENSIVE METRICS")
    print("=" * 60)
    
    # Check if segmentation model exists
    if not os.path.exists(Config.segmentation_model_path):
        print(f"Error: Segmentation model not found at {Config.segmentation_model_path}")
        print("Please train the segmentation model first using main.py")
        return
    
    # Load model
    print("Loading segmentation model...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(Config.segmentation_model_path, map_location=device))
    model.eval()
    
    # Get test images
    test_image_paths = sorted(glob.glob(os.path.join(Config.test_image_dir, "*.png")))
    
    # Initialize evaluation metrics
    total_dice = 0.0
    total_accuracy = 0.0
    total_seg_precision = 0.0
    total_seg_recall = 0.0
    total_seg_f1 = 0.0
    
    # Detection metrics
    total_det_precision = 0.0
    total_det_recall = 0.0
    total_det_f1 = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_boxes = 0
    
    # For mAP calculation
    all_predictions = {}
    all_ground_truths = {}
    
    processed_count = 0
    images_with_seg_gt = 0
    
    print("\nAnalyzing images with ground truth labels...")
    
    for image_path in test_image_paths:
        base_name = os.path.basename(image_path).replace(".png", "")
        label_path = os.path.join(Config.test_label_dir, f"{base_name}.txt")
        
        # Only analyze images that have ground truth
        if not os.path.exists(label_path):
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        actual_height, actual_width = image.shape[:2]
        
        # Preprocess for model (use 256x640 as in training)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model_image = cv2.resize(image_rgb, (640, 256))  # Width, Height
        model_image = model_image.astype(np.float32) / 255.0
        model_image = np.transpose(model_image, (2, 0, 1))
        model_tensor = torch.FloatTensor(model_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(model_tensor)
        
        # Get output statistics
        output_np = output.cpu().numpy()[0, 0]
        
        # Resize to original size
        mask_resized = cv2.resize(output_np, (actual_width, actual_height))
        
        # Create binary mask for evaluation
        binary_mask = (mask_resized > Config.confidence_threshold).astype(np.uint8)
        
        # Load ground truth mask if available
        gt_mask_path = os.path.join(Config.data_root, "test/ground_truth", f"{base_name}_GT.png")
        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 0).astype(np.uint8)
            
            # Calculate segmentation metrics
            dice = calculate_dice_coefficient(binary_mask, gt_mask)
            accuracy = calculate_pixel_accuracy(binary_mask, gt_mask)
            seg_precision, seg_recall, seg_f1 = calculate_precision_recall_f1(binary_mask, gt_mask)
            
            total_dice += dice
            total_accuracy += accuracy
            total_seg_precision += seg_precision
            total_seg_recall += seg_recall
            total_seg_f1 += seg_f1
            images_with_seg_gt += 1
        
        # Find contours for detection
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes from predictions with confidence scores
        pred_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= Config.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence from the mask (max value in the region)
                roi_mask = mask_resized[y:y+h, x:x+w]
                confidence = roi_mask.max() if roi_mask.size > 0 else Config.confidence_threshold
                
                pred_boxes.append([x, y, x + w, y + h, confidence])
        
        # Load ground truth boxes
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_center *= actual_width
                    y_center *= actual_height
                    width *= actual_width
                    height *= actual_height
                    
                    x_min = int(x_center - width/2)
                    y_min = int(y_center - height/2)
                    x_max = int(x_center + width/2)
                    y_max = int(y_center + height/2)
                    
                    gt_boxes.append([x_min, y_min, x_max, y_max])
        
        # Store predictions and ground truths for mAP calculation
        all_predictions[base_name] = pred_boxes
        all_ground_truths[base_name] = gt_boxes
        
        # Calculate detection metrics for current image
        det_precision, det_recall, det_f1, tp, fp, fn = evaluate_detection(
            [box[:4] for box in pred_boxes], gt_boxes
        )
        
        total_det_precision += det_precision
        total_det_recall += det_recall
        total_det_f1 += det_f1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_boxes += len(gt_boxes)
        processed_count += 1
        
        # Progress update
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count} images...")
    
    # 確保有處理過的圖片
    if processed_count == 0:
        print("\nNo images with ground truth found for evaluation.")
        return
    
    # Calculate mAP
    mAP_results = {}
    for iou_thresh in Config.iou_thresholds:
        ap = calculate_map(all_predictions, all_ground_truths, iou_thresh)
        mAP_results[f'AP@{iou_thresh:.2f}'] = ap
    
    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 60)
    print(f"Total images evaluated: {processed_count}")
    print(f"Images with segmentation ground truth: {images_with_seg_gt}")
    
    # 計算整體檢測指標（從彙總的TP/FP/FN）
    overall_det_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_det_recall = total_tp / total_gt_boxes if total_gt_boxes > 0 else 0.0
    overall_det_f1 = 2 * overall_det_precision * overall_det_recall / (overall_det_precision + overall_det_recall) if (overall_det_precision + overall_det_recall) > 0 else 0.0
    
    # 計算平均每圖檢測指標
    avg_det_precision = total_det_precision / processed_count
    avg_det_recall = total_det_recall / processed_count
    avg_det_f1 = total_det_f1 / processed_count
    
    print("\n" + "-" * 40)
    print("DETECTION METRICS")
    print("-" * 40)
    
    print("\nOverall Detection Metrics (from aggregated statistics):")
    print(f"  Precision: {overall_det_precision:.4f}")
    print(f"  Recall: {overall_det_recall:.4f}")
    print(f"  F1-Score: {overall_det_f1:.4f}")
    print(f"  True Positives: {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Total Ground Truth Objects: {total_gt_boxes}")
    
    print(f"\nAverage per-image Detection Metrics (IoU=0.5):")
    print(f"  Precision: {avg_det_precision:.4f}")
    print(f"  Recall: {avg_det_recall:.4f}")
    print(f"  F1-Score: {avg_det_f1:.4f}")
    
    print(f"\nmAP Metrics:")
    for key, value in mAP_results.items():
        print(f"  {key}: {value:.4f}")
    
    # 計算平均mAP（如果有多個閾值）
    if len(mAP_results) > 0:
        avg_mAP = np.mean(list(mAP_results.values()))
        print(f"  Average mAP: {avg_mAP:.4f}")
    
    # 分割指標（如果有）
    if images_with_seg_gt > 0:
        print("\n" + "-" * 40)
        print("SEGMENTATION METRICS (Average)")
        print("-" * 40)
        
        avg_dice = total_dice / images_with_seg_gt
        avg_accuracy = total_accuracy / images_with_seg_gt
        avg_seg_precision = total_seg_precision / images_with_seg_gt
        avg_seg_recall = total_seg_recall / images_with_seg_gt
        avg_seg_f1 = total_seg_f1 / images_with_seg_gt
        
        print(f"  Dice Coefficient: {avg_dice:.4f}")
        print(f"  Pixel Accuracy: {avg_accuracy:.4f}")
        print(f"  Precision: {avg_seg_precision:.4f}")
        print(f"  Recall: {avg_seg_recall:.4f}")
        print(f"  F1-Score: {avg_seg_f1:.4f}")
    
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    
    # 創建摘要表格
    print("\n{:<25} {:<15} {:<15}".format("Metric Type", "Metric", "Value"))
    print("-" * 55)
    
    # 檢測指標
    print("{:<25} {:<15} {:<15.4f}".format("Detection", "Precision", overall_det_precision))
    print("{:<25} {:<15} {:<15.4f}".format("Detection", "Recall", overall_det_recall))
    print("{:<25} {:<15} {:<15.4f}".format("Detection", "F1-Score", overall_det_f1))
    
    # mAP指標
    for key, value in mAP_results.items():
        print("{:<25} {:<15} {:<15.4f}".format("Detection", key, value))
    
    if len(mAP_results) > 0:
        avg_mAP = np.mean(list(mAP_results.values()))
        print("{:<25} {:<15} {:<15.4f}".format("Detection", "Average mAP", avg_mAP))
    
    # 分割指標（如果有）
    if images_with_seg_gt > 0:
        avg_dice = total_dice / images_with_seg_gt
        avg_accuracy = total_accuracy / images_with_seg_gt
        avg_seg_precision = total_seg_precision / images_with_seg_gt
        avg_seg_recall = total_seg_recall / images_with_seg_gt
        avg_seg_f1 = total_seg_f1 / images_with_seg_gt
        
        print("{:<25} {:<15} {:<15.4f}".format("Segmentation", "Dice Coefficient", avg_dice))
        print("{:<25} {:<15} {:<15.4f}".format("Segmentation", "Pixel Accuracy", avg_accuracy))
        print("{:<25} {:<15} {:<15.4f}".format("Segmentation", "Precision", avg_seg_precision))
        print("{:<25} {:<15} {:<15.4f}".format("Segmentation", "Recall", avg_seg_recall))
        print("{:<25} {:<15} {:<15.4f}".format("Segmentation", "F1-Score", avg_seg_f1))
    
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Confidence threshold: {Config.confidence_threshold}")
    print(f"Minimum area: {Config.min_area}")
    print(f"IoU thresholds for mAP: {Config.iou_thresholds}")
    
    # Save overall results to file
    results_file = "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("KSDD2 Dataset Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total images evaluated: {processed_count}\n")
        f.write(f"Images with segmentation GT: {images_with_seg_gt}\n")
        f.write(f"Confidence threshold: {Config.confidence_threshold}\n")
        f.write(f"Minimum area: {Config.min_area}\n")
        f.write(f"IoU thresholds for mAP: {Config.iou_thresholds}\n")
        
        f.write("\nDETECTION METRICS:\n")
        f.write(f"  Overall Precision: {overall_det_precision:.4f}\n")
        f.write(f"  Overall Recall: {overall_det_recall:.4f}\n")
        f.write(f"  Overall F1-Score: {overall_det_f1:.4f}\n")
        f.write(f"  True Positives: {total_tp}\n")
        f.write(f"  False Positives: {total_fp}\n")
        f.write(f"  False Negatives: {total_fn}\n")
        f.write(f"  Total Ground Truth: {total_gt_boxes}\n")
        
        f.write(f"\n  Average per-image metrics:\n")
        f.write(f"    Precision: {avg_det_precision:.4f}\n")
        f.write(f"    Recall: {avg_det_recall:.4f}\n")
        f.write(f"    F1-Score: {avg_det_f1:.4f}\n")
        
        f.write(f"\n  mAP METRICS:\n")
        for key, value in mAP_results.items():
            f.write(f"    {key}: {value:.4f}\n")
        
        if len(mAP_results) > 0:
            avg_mAP = np.mean(list(mAP_results.values()))
            f.write(f"    Average mAP: {avg_mAP:.4f}\n")
        
        if images_with_seg_gt > 0:
            f.write("\nSEGMENTATION METRICS (Average):\n")
            f.write(f"  Dice Coefficient: {total_dice/images_with_seg_gt:.4f}\n")
            f.write(f"  Pixel Accuracy: {total_accuracy/images_with_seg_gt:.4f}\n")
            f.write(f"  Precision: {total_seg_precision/images_with_seg_gt:.4f}\n")
            f.write(f"  Recall: {total_seg_recall/images_with_seg_gt:.4f}\n")
            f.write(f"  F1-Score: {total_seg_f1/images_with_seg_gt:.4f}\n")
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Save detailed results in JSON format
    detailed_results = {
        "config": {
            "confidence_threshold": float(Config.confidence_threshold),
            "min_area": int(Config.min_area),
            "iou_thresholds": [float(t) for t in Config.iou_thresholds]
        },
        "dataset_stats": {
            "processed_images": int(processed_count),
            "images_with_segmentation_gt": int(images_with_seg_gt)
        },
        "detection_metrics": {
            "overall": {
                "precision": float(overall_det_precision),
                "recall": float(overall_det_recall),
                "f1_score": float(overall_det_f1),
                "true_positives": int(total_tp),
                "false_positives": int(total_fp),
                "false_negatives": int(total_fn),
                "total_ground_truth": int(total_gt_boxes)
            },
            "average_per_image": {
                "precision": float(avg_det_precision),
                "recall": float(avg_det_recall),
                "f1_score": float(avg_det_f1)
            },
            "mAP": mAP_results
        }
    }
    
    if images_with_seg_gt > 0:
        detailed_results["segmentation_metrics"] = {
            "average": {
                "dice_coefficient": float(total_dice/images_with_seg_gt),
                "pixel_accuracy": float(total_accuracy/images_with_seg_gt),
                "precision": float(total_seg_precision/images_with_seg_gt),
                "recall": float(total_seg_recall/images_with_seg_gt),
                "f1_score": float(total_seg_f1/images_with_seg_gt)
            }
        }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    detailed_results = convert_to_serializable(detailed_results)
    
    with open("detailed_evaluation_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Detailed JSON results saved to detailed_evaluation_results.json")

def visualize_predictions():
    """Visualize predictions with different thresholds"""
    print("\n" + "=" * 60)
    print("PREDICTION VISUALIZATION")
    print("=" * 60)
    
    # Check if segmentation model exists
    if not os.path.exists(Config.segmentation_model_path):
        print(f"Error: Segmentation model not found at {Config.segmentation_model_path}")
        return
    
    # Load model
    print("Loading segmentation model...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(Config.segmentation_model_path, map_location=device))
    model.eval()
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Get test images with ground truth
    test_image_paths = sorted(glob.glob(os.path.join(Config.test_image_dir, "*.png")))
    
    # Process images with ground truth
    processed = 0
    for image_path in test_image_paths:
        if processed >= 20:  # Limit to 20 images
            break
            
        base_name = os.path.basename(image_path).replace(".png", "")
        label_path = os.path.join(Config.test_label_dir, f"{base_name}.txt")
        
        # Only process images with ground truth
        if not os.path.exists(label_path):
            continue
        
        processed += 1
        print(f"\nProcessing {processed}/20: {base_name}.png")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        actual_height, actual_width = image.shape[:2]
        
        # Preprocess for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model_image = cv2.resize(image_rgb, (640, 256))
        model_image = model_image.astype(np.float32) / 255.0
        model_image = np.transpose(model_image, (2, 0, 1))
        model_tensor = torch.FloatTensor(model_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(model_tensor)
        
        # Get output and resize
        output_np = output.cpu().numpy()[0, 0]
        mask_resized = cv2.resize(output_np, (actual_width, actual_height))
        
        # Try multiple thresholds
        thresholds = [0.1, 0.3, 0.5]
        
        # Load ground truth
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_center *= actual_width
                    y_center *= actual_height
                    width *= actual_width
                    height *= actual_height
                    
                    x_min = int(x_center - width/2)
                    y_min = int(y_center - height/2)
                    x_max = int(x_center + width/2)
                    y_max = int(y_center + height/2)
                    
                    gt_boxes.append([x_min, y_min, x_max, y_max])
        
        # Create visualization for each threshold
        for threshold in thresholds:
            # Convert mask to binary
            binary_mask = (mask_resized > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get bounding boxes with confidence values
            pred_boxes_with_conf = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= Config.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence from the mask
                    roi_mask = mask_resized[y:y+h, x:x+w]
                    max_confidence = roi_mask.max()
                    
                    pred_boxes_with_conf.append([x, y, x + w, y + h, max_confidence])
            
            # Create combined visualization
            combined = image.copy()
            
            # Draw ground truth in red
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(combined, "GT", (x1, max(y1 - 10, 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw predictions in green with confidence numbers
            for box_info in pred_boxes_with_conf:
                x1, y1, x2, y2, max_conf = box_info
                
                # Draw bounding box
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare confidence text
                conf_text = f"conf:{max_conf:.2f}"
                
                # Calculate text position
                text_y = max(y1 - 10, 20)  # Position above the box, minimum 20 pixels from top
                
                # Draw text background for better visibility
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(combined, (x1, text_y - text_size[1] - 4), 
                            (x1 + text_size[0] + 4, text_y + 4), (0, 255, 0), -1)
                
                # Draw confidence text
                cv2.putText(combined, conf_text, (x1 + 2, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text
            
            # Add mask overlay
            mask_display = (mask_resized * 255).astype(np.uint8)
            if mask_display.max() > 0:
                green_overlay = np.zeros_like(image)
                green_overlay[:, :, 1] = (mask_resized > threshold) * 255
                combined = cv2.addWeighted(combined, 0.7, green_overlay, 0.3, 0)
            
            # Calculate detection metrics (只取前三個值，忽略tp, fp, fn)
            pred_boxes = [box[:4] for box in pred_boxes_with_conf]  # Extract just coordinates for evaluation
            det_precision, det_recall, det_f1, _, _, _ = evaluate_detection(pred_boxes, gt_boxes)
            
            # Save result
            cv2.imwrite(os.path.join(Config.output_dir, f"{base_name}_thresh{threshold}.png"), combined)
            
            print(f"  Threshold {threshold}: {len(pred_boxes)} predictions")
            print(f"    Precision: {det_precision:.4f}, Recall: {det_recall:.4f}, F1: {det_f1:.4f}")

def check_training_results():
    """Check if the model was properly trained"""
    print("\n" + "=" * 60)
    print("TRAINING RESULTS CHECK")
    print("=" * 60)
    
    # Check model files
    print("Checking model files...")
    if os.path.exists("segmentation_model.pth"):
        print("✓ segmentation_model.pth exists")
        file_size = os.path.getsize("segmentation_model.pth") / 1024 / 1024
        print(f"  File size: {file_size:.2f} MB")
    else:
        print("✗ segmentation_model.pth NOT FOUND")
        
    if os.path.exists("detection_model.pth"):
        print("✓ detection_model.pth exists")
        file_size = os.path.getsize("detection_model.pth") / 1024 / 1024
        print(f"  File size: {file_size:.2f} MB")
    else:
        print("✗ detection_model.pth NOT FOUND")
    
    # Check training logs if they exist
    print("\nChecking sample predictions on training data...")
    
    # Try to load and test on one training image
    train_image_dir = os.path.join(Config.data_root, "train/image")
    train_label_dir = os.path.join(Config.data_root, "train/label")
    
    if os.path.exists(train_image_dir):
        train_images = sorted(glob.glob(os.path.join(train_image_dir, "*.png")))[:5]
        
        for img_path in train_images:
            base_name = os.path.basename(img_path).replace(".png", "")
            label_path = os.path.join(train_label_dir, f"{base_name}.txt")
            
            has_label = os.path.exists(label_path)
            print(f"  {base_name}.png: {'Has label' if has_label else 'No label'}")

# Main function
def main():
    print("=" * 60)
    print("KSDD2 DETECTION DEBUGGER WITH COMPREHENSIVE METRICS")
    print("=" * 60)
    
    # Check data exists
    if not os.path.exists(Config.data_root):
        print(f"Error: Data directory not found at {Config.data_root}")
        return
    
    # Run checks
    check_training_results()
    analyze_model_output()  # This now includes mAP and comprehensive metrics
    visualize_predictions()
    
    print("\n" + "=" * 60)
    print("DEBUGGING AND EVALUATION COMPLETE")
    print("=" * 60)
    print("\nResults summary:")
    print("1. Check evaluation_results.txt for overall metrics")
    print("2. Check detailed_evaluation_results.json for detailed metrics")
    print("3. Look at detection_results/ folder for visualizations")
    print("4. Check debug_analysis/ folder for debug images")
    print("\nMetrics calculated include:")
    print("  - mAP (mean Average Precision)")
    print("  - Precision, Recall, F1-Score for detection")
    print("  - Dice Coefficient for segmentation")
    print("  - Pixel Accuracy for segmentation")
    print("  - Precision, Recall, F1-Score for segmentation")

if __name__ == "__main__":
    main()