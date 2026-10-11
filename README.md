# Road-D-Loc: Road Defect Localization and Quantification

**Author:** Tarun Yadav  
**Registration Number:** 24BAI10576  
**License:** Apache 2.0  

---

## 1. Project Overview

Traditional road inspection methods rely on manual, time-consuming visual surveys that are hazardous, costly, and error-prone. **Road-D-Loc** is an automated computer vision system designed to inspect road surfaces using deep learning-based semantic segmentation. It rapidly assesses road surfaces from static digital images to:

1. **Detect and Localize Defects:** Distinguish between **Potholes** and **Cracks** at the pixel level.
2. **Quantify Severity:** Measure defect area in square pixels and percentage of road surface.
3. **Generate Actionable Reports:** Produce structured CSV reports (`Defect_Report.csv`) and visual overlays for maintenance planning and infrastructure management.

---

## 2. System Architecture

The pipeline consists of three core functional modules:

```
[ Input Road Image ]
        │
        ▼
┌────────────────────────────────────────┐
│ Module 1: Preprocessing & Acquisition  │
│ (cv2 / torchvision transforms)         │
│ • Resizing to 256x256                  │
│ • BGR to RGB conversion                │
│ • ImageNet normalization               │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│ Module 2: Deep Segmentation (U-Net)    │
│ • 4-level Convolutional Encoder        │
│ • Bottleneck Feature Representation    │
│ • Decoder with Skip Connections        │
│ • 3-Class Pixel Segmentation:          │
│   - Class 0: Background                │
│   - Class 1: Pothole                   │
│   - Class 2: Crack                     │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│ Module 3: Postprocessing & Reporting   │
│ • Tensor to argmax mask                │
│ • Resize mask to original dimensions   │
│ • Defect pixel count & area percentage │
│ • Severity categorization              │
│   (LOW / MEDIUM / HIGH)                │
│ • Alpha-blended visual overlay         │
│ • Defect_Report.csv export             │
└────────────────────────────────────────┘
        │
        ├──► Segmented Overlay Image (output/)
        └──► Defect_Report.csv (data/)
```

---

## 3. Directory Structure

```
.
├── main.py                    # Main pipeline orchestrator and CLI entry point
├── requirements.txt           # Project dependencies
├── statement.md               # Detailed problem statement and scope
├── LICENSE                    # Apache 2.0 License
├── README.md                  # Project documentation
├── models/
│   ├── __init__.py            # Model package initializer
│   ├── model_architecture.py  # PyTorch U-Net neural network architecture
│   └── checkpoints/           # Directory for model weight checkpoints (.pth)
├── src/
│   ├── __init__.py            # Source package initializer
│   ├── preprocess.py          # Module 1: Image loading, resizing, normalization
│   └── postprocess_report.py  # Module 3: Mask generation, area calculation, CSV logging
├── data/
│   └── Defect_Report.csv      # Generated quantitative defect logs
└── output/
    └── *_segmented.jpg        # Overlay images highlighting defects
```

---

## 4. Installation & Requirements

Ensure you have Python 3.9+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies:
- `torch` (PyTorch)
- `torchvision`
- `opencv-python`
- `numpy`
- `pandas`

---

## 5. Usage & Execution

### Basic Run (Demonstration / Test Bench)
If no image is provided, the system automatically creates a realistic synthetic test bench image and performs defect segmentation:

```bash
python main.py
```

### Run with a Custom Image
```bash
python main.py path/to/road_image.jpg
```

### Run with Trained Model Weights
```bash
python main.py path/to/road_image.jpg --weights ./models/checkpoints/best_unet_weights.pth --output ./output --report ./data/Defect_Report.csv
```

---

## 6. Output & Defect Reporting

### 1. Defect Report (`data/Defect_Report.csv`)
Logs quantitative metrics for each detected defect:
- `timestamp`: Date and time of processing.
- `inspector_name`: Tarun Yadav
- `registration_no`: 24BAI10576
- `image_id`: Source image filename.
- `defect_type`: "Pothole" or "Crack".
- `area_sq_pixels`: Total segmented defect pixel count.
- `area_percent`: Percentage of total road surface occupied by defect.
- `severity`: Assessment based on defect area:
  - **LOW**: $\le 0.5\%$
  - **MEDIUM**: $0.5\% - 2.0\%$
  - **HIGH**: $> 2.0\%$

### 2. Visual Segmentation Overlay
Visual outputs are saved in `./output/*_segmented.jpg` with color-coded highlights:
- **Red (BGR: 0, 0, 255):** Potholes
- **Green (BGR: 0, 255, 0):** Cracks
- **Alpha Blending:** 50% opacity preserves the underlying road texture for verification.