"""
Project: Road-D-Loc (Road Defect Localization and Quantification)
Author: Tarun Yadav
Registration No: 24BAI10576
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch

# Ensure current directory is in sys.path for robust module resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the functional modules
from models.model_architecture import load_model, UNet
from src.preprocess import load_and_preprocess
from src.postprocess_report import create_segmentation_mask, quantify_and_report, visualize_mask


# --- Mock Data Simulation ---
def mock_prediction(input_tensor, num_classes=3):
    """
    Mocks the output of the U-Net model for demonstration.
    Classes:
      0: Background
      1: Pothole
      2: Crack
    """
    B, C, H, W = input_tensor.shape
    
    # Initialize background logits higher than defects
    mock_raw_output = torch.zeros((B, num_classes, H, W), dtype=torch.float32)
    mock_raw_output[:, 0, :, :] = 1.0  # Background default
    
    # Inject a simulated defect in the center-right area for Class 1 (Pothole)
    center_h, center_w = H // 2, W // 2
    mock_raw_output[0, 1, center_h-15:center_h+15, center_w+20:center_w+55] = 10.0 
    
    # Inject a simulated defect in the bottom-left area for Class 2 (Crack)
    mock_raw_output[0, 2, center_h+30:center_h+60, center_w-60:center_w-15] = 10.0 
    
    return mock_raw_output


# --- Main Pipeline ---
def run_road_d_loc(image_path, model_weights_path="./models/checkpoints/best_unet_weights.pth", output_dir="output", report_path="./data/Defect_Report.csv"):
    """
    The main execution flow, demonstrating the logical workflow.
    """
    print("=" * 55)
    print(" Road-D-Loc: Road Defect Localization & Quantification")
    print(" Author: Tarun Yadav | Reg No: 24BAI10576")
    print("=" * 55)

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    report_dir = os.path.dirname(os.path.abspath(report_path))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    
    # 1. Module 1: Image Acquisition and Preprocessing
    try:
        original_img, input_tensor = load_and_preprocess(image_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Determine original size for resizing the mask back
    original_h, original_w = original_img.shape[:2]

    # 2. Module 2: Deep Segmentation and Localization
    if os.path.exists(model_weights_path):
        print(f"INFO: Loading pre-trained weights from {model_weights_path}...")
        model = load_model(model_weights_path, num_classes=3)
        with torch.no_grad():
            prediction_tensor = model(input_tensor)
        print("INFO: Performing Neural Network Segmentation Inference...")
    else:
        print("INFO: No weights file found at specified path.")
        print("INFO: Running demonstration mode with simulated U-Net localization...")
        prediction_tensor = mock_prediction(input_tensor, num_classes=3)

    # 3. Post-processing: Convert tensor to mask
    mask_resized = create_segmentation_mask(prediction_tensor, (original_w, original_h))

    # 4. Module 3: Defect Quantification and Reporting
    image_name = os.path.basename(image_path)
    quantify_and_report(mask_resized, image_name, report_path=report_path, inspector="Tarun Yadav", reg_no="24BAI10576")

    # 5. Visualization (part of Module 3 output)
    name_part, ext = os.path.splitext(image_name)
    output_filename = f"{name_part}_segmented{ext if ext else '.jpg'}"
    output_path = os.path.join(output_dir, output_filename)
    visualize_mask(original_img, mask_resized, output_path)

    print("\n--- Processing Complete ---")
    print(f"Segmented Image saved to: {output_path}")
    print(f"Defect Report saved to:   {report_path}")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Road Surface Defect Localization System (Road-D-Loc) - Tarun Yadav (24BAI10576)")
    parser.add_argument("image_path", type=str, nargs="?", default="test_road.jpg", help="Path to the input road image.")
    parser.add_argument("--weights", type=str, default="./models/checkpoints/best_unet_weights.pth", 
                        help="Path to the pre-trained U-Net weights file.")
    parser.add_argument("--output", type=str, default="./output", help="Directory to save segmented outputs.")
    parser.add_argument("--report", type=str, default="./data/Defect_Report.csv", help="Path to defect report CSV.")
    
    args = parser.parse_args()

    # --- Setup Environment ---
    os.makedirs("./models/checkpoints", exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    # If the user doesn't provide an image or test image does not exist, create a realistic road image.
    if not os.path.exists(args.image_path):
        dummy_img = np.zeros((500, 800, 3), dtype=np.uint8) + 110 # Asphalt gray
        # Add road lane markings
        cv2.line(dummy_img, (0, 470), (800, 470), (255, 255, 255), 4)
        cv2.line(dummy_img, (0, 30), (800, 30), (255, 255, 255), 4)
        for x in range(40, 760, 100):
            cv2.line(dummy_img, (x, 250), (x+50, 250), (0, 255, 255), 4) # Yellow dashed center line
        cv2.putText(dummy_img, "ROAD-D-LOC TEST BENCH", (180, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(dummy_img, "Inspector: Tarun Yadav (24BAI10576)", (160, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.imwrite("test_road.jpg", dummy_img)
        args.image_path = "test_road.jpg"
        print(f"NOTE: Created a test road image '{args.image_path}' for demonstration.")
    
    run_road_d_loc(args.image_path, args.weights, output_dir=args.output, report_path=args.report)