"""
Project: Road-D-Loc (Road Defect Localization and Quantification)
Module 3: Defect Quantification and Reporting
Author: Tarun Yadav
Registration No: 24BAI10576
"""

import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import torch

# Define colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 0),       # Background (Black / transparent)
    1: (0, 0, 255),     # Class 1: Pothole (Red)
    2: (0, 255, 0)      # Class 2: Crack (Green)
}

CLASS_NAMES = {
    1: "Pothole",
    2: "Crack"
}

def create_segmentation_mask(prediction_tensor, target_size):
    """
    Converts model output tensor to a NumPy segmentation mask resized to target_size (W, H).
    """
    # If it's a PyTorch tensor, extract argmax across channel dimension
    if isinstance(prediction_tensor, torch.Tensor):
        if prediction_tensor.dim() == 4:
            # (B, C, H, W) -> argmax over dim=1
            mask_tensor = torch.argmax(prediction_tensor, dim=1).squeeze(0).cpu().numpy()
        elif prediction_tensor.dim() == 3:
            # (C, H, W) -> argmax over dim=0
            mask_tensor = torch.argmax(prediction_tensor, dim=0).cpu().numpy()
        else:
            mask_tensor = prediction_tensor.cpu().numpy()
    else:
        # NumPy array fallback
        if prediction_tensor.ndim == 4:
            mask_tensor = np.argmax(prediction_tensor, axis=1)[0]
        elif prediction_tensor.ndim == 3:
            mask_tensor = np.argmax(prediction_tensor, axis=0)
        else:
            mask_tensor = prediction_tensor

    # Resize mask back to the original target resolution for accurate calculation
    mask_resized = cv2.resize(mask_tensor.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
    
    return mask_resized

def quantify_and_report(mask_resized, image_name, report_path="./data/Defect_Report.csv", inspector="Tarun Yadav", reg_no="24BAI10576"):
    """
    Module 3: Calculates defect area and logs the report.
    """
    results = []
    total_pixels = mask_resized.size
    
    print("\nINFO: Quantifying defects...")
    
    for class_id, class_name in CLASS_NAMES.items():
        # Count pixels belonging to the current class ID
        defect_pixels = int(np.sum(mask_resized == class_id))
        
        # Calculate area percentage
        area_percentage = (defect_pixels / total_pixels) * 100
        
        # Simple severity assignment based on area
        severity = "LOW"
        if area_percentage > 0.5:
            severity = "MEDIUM"
        if area_percentage > 2.0:
            severity = "HIGH"
            
        if defect_pixels > 0:
            record = {
                'timestamp': datetime.now().isoformat(),
                'inspector_name': inspector,
                'registration_no': reg_no,
                'image_id': image_name,
                'defect_type': class_name,
                'area_sq_pixels': defect_pixels,
                'area_percent': round(area_percentage, 4),
                'severity': severity
            }
            results.append(record)
            print(f"  - Found {class_name}: {defect_pixels} pixels ({area_percentage:.4f}%) - Severity: {severity}")
            
    # Log data to CSV (NFR: Data integrity/Logging)
    if results:
        # Ensure target directory exists before saving
        report_dir = os.path.dirname(os.path.abspath(report_path))
        os.makedirs(report_dir, exist_ok=True)

        df = pd.DataFrame(results)
        
        if not os.path.exists(report_path):
            df.to_csv(report_path, index=False)
        else:
            df.to_csv(report_path, mode='a', header=False, index=False)
        
        print(f"INFO: Defect report logged successfully to {report_path}")

    return results

def visualize_mask(original_img, mask_resized, output_path):
    """
    Overlay the segmented mask onto the original image for visualization.
    """
    # Create an empty overlay image
    overlay = np.zeros_like(original_img, dtype=np.uint8)
    
    has_defects = False
    for class_id, color in CLASS_COLORS.items():
        if class_id == 0:
            continue
        # Apply color to the pixels of the corresponding class
        mask_class = (mask_resized == class_id)
        if np.any(mask_class):
            has_defects = True
            overlay[mask_class] = color
        
    # Blend the original image with the mask (alpha blending)
    if has_defects:
        alpha = 0.5 
        segmented_img = cv2.addWeighted(original_img, 1.0, overlay, alpha, 0)
    else:
        segmented_img = original_img.copy()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the output
    cv2.imwrite(output_path, segmented_img)
    print(f"INFO: Segmented image saved to {output_path}")
