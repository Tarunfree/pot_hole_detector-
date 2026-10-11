"""
Compatibility redirect for postprocess_report module.
Author: Tarun Yadav (24BAI10576)
"""
from src.postprocess_report import (
    CLASS_COLORS,
    CLASS_NAMES,
    create_segmentation_mask,
    quantify_and_report,
    visualize_mask
)

__all__ = [
    'CLASS_COLORS',
    'CLASS_NAMES',
    'create_segmentation_mask',
    'quantify_and_report',
    'visualize_mask'
]