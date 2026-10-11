"""
Unit Tests for Road-D-Loc Pipeline
Author: Tarun Yadav
Registration No: 24BAI10576
"""

import unittest
import os
import numpy as np
import torch
import cv2

from models.model_architecture import UNet, load_model
from src.preprocess import load_and_preprocess
from src.postprocess_report import create_segmentation_mask, quantify_and_report, visualize_mask

class TestRoadDLocPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary test image
        self.test_img_path = "test_unit.jpg"
        img = np.zeros((200, 300, 3), dtype=np.uint8) + 128
        cv2.imwrite(self.test_img_path, img)
        self.output_dir = "test_output"
        self.report_path = "test_output/test_report.csv"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
        if os.path.exists(self.report_path):
            os.remove(self.report_path)
        if os.path.exists(self.output_dir):
            import shutil
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_module_1_preprocessing(self):
        orig_img, tensor = load_and_preprocess(self.test_img_path, target_size=(256, 256))
        self.assertEqual(orig_img.shape, (200, 300, 3))
        self.assertEqual(tensor.shape, (1, 3, 256, 256))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_module_2_unet_architecture(self):
        model = UNet(in_channels=3, out_channels=3)
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 3, 256, 256))

    def test_module_3_postprocessing_and_report(self):
        # Create dummy prediction tensor (1, 3, 100, 100)
        pred = torch.zeros((1, 3, 100, 100))
        pred[0, 1, 10:20, 10:20] = 5.0 # Pothole
        pred[0, 2, 40:60, 40:60] = 5.0 # Crack
        
        mask = create_segmentation_mask(pred, (300, 200))
        self.assertEqual(mask.shape, (200, 300))
        
        results = quantify_and_report(mask, "test_unit.jpg", report_path=self.report_path)
        self.assertTrue(len(results) >= 2)
        self.assertTrue(os.path.exists(self.report_path))
        
        # Verify columns in generated CSV
        import pandas as pd
        df = pd.read_csv(self.report_path)
        self.assertIn('inspector_name', df.columns)
        self.assertIn('registration_no', df.columns)
        self.assertEqual(df['inspector_name'].iloc[0], "Tarun Yadav")
        self.assertEqual(df['registration_no'].iloc[0], "24BAI10576")

if __name__ == '__main__':
    unittest.main()
