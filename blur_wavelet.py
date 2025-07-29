#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Haar Wavelet Blur Detection with Feature Density Analysis

Backward compatibility wrapper - imports from refactored modules.
For new projects, use main.py directly.

@author: pedrofRodenas (original), enhanced with feature density analysis
"""

# Import all functions from the refactored modules for backward compatibility
from detector import (
    blur_detect,
    advanced_blur_detect,
    extract_center_region,
    calculate_feature_density,
    is_low_feature_image
)

from utils import (
    calculate_quality_score,
    get_quality_color,
    display_intermediate_results,
    create_false_cases_display,
    find_images,
    setup_window
)

# Import main function for direct execution
from main import main

# For backward compatibility, ensure this file can still be run directly
if __name__ == '__main__':
    print("🔄 Running refactored blur detection system...")
    print("💡 For new projects, consider using 'python main.py' directly.")
    print()
    main()







