#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Haar Wavelet Blur Detection with Feature Density Analysis

Main entry point for the blur detection application.
"""

import cv2
import os
import argparse
import json

# Import our modules
from detector import advanced_blur_detect
from utils import (
    display_intermediate_results, 
    create_false_cases_display, 
    find_images, 
    setup_window
)


def main():
    """Main function for blur detection application"""
    parser = argparse.ArgumentParser(
        description='Advanced Haar Wavelet blur detection with feature density analysis',
        epilog='''
Examples:
  python main.py -i images/                                    # Basic usage
  python main.py -i images/ --window-size 1600x1000           # Custom window size  
  python main.py -i images/ --show-false-cases sharp          # Show misclassified cases
  python main.py -i images/ --conservative-threshold 0.8      # Stricter for low-feature images
  
Display Layout:
  - Main image with quality score prominently displayed on left
  - Compact edge detection results in sidebar on right  
  - Center region outlined in yellow for analysis visualization
  - High-quality antialiased text rendering (50% smaller, crisper appearance)
  - Adaptive text sizing prevents cut-off at different window sizes
  
Window Controls:
  - Drag window edges to resize
  - Press any key to continue, 'q' to quit
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input_dir', dest="input_dir", type=str, required=True, 
                       help="directory of images")
    parser.add_argument('-s', '--save_path', dest='save_path', type=str, 
                       help="path to save output")
    parser.add_argument("-t", "--threshold", dest='threshold', type=float, default=35, 
                       help="blurry threshold")
    parser.add_argument("-d", "--decision", dest='MinZero', type=float, default=0.001, 
                       help="MinZero Decision Threshold")
    parser.add_argument("--no-display", dest='no_display', action='store_true', 
                       help="disable interactive display")
    parser.add_argument("--show-false-cases", dest='show_false_cases', choices=['blur', 'sharp'], 
                       help="display false classification cases - specify expected result: 'blur' or 'sharp'")
    parser.add_argument("--conservative-threshold", dest='conservative_threshold', type=float, default=0.7, 
                       help="blur threshold for low-feature images (default: 0.7)")
    parser.add_argument("--feature-sensitivity", dest='feature_sensitivity', type=float, default=0.75, 
                       help="sensitivity for low-feature detection (0.5-1.0, default: 0.75)")
    parser.add_argument("--window-size", dest='window_size', type=str, default="1200x800", 
                       help="initial window size in WIDTHxHEIGHT format (default: 1200x800)")
    
    args = parser.parse_args()
    
    # Initialize variables
    results = []
    false_cases = []  # Track false classification cases
    expected_classification = None  # User-specified expected result
    
    # Set expected classification if false case detection is enabled
    if args.show_false_cases:
        expected_classification = args.show_false_cases == 'blur'  # True for blur, False for sharp
        print(f"📋 False case detection enabled: All images expected to be '{args.show_false_cases.upper()}'")
        print(f"Will identify images incorrectly classified as '{'SHARP' if expected_classification else 'BLUR'}'\n")
    
    # Create window if display is enabled
    window_name = 'Advanced Blur Detection - Main Focus | Edge Sidebar'
    if not args.no_display:
        setup_window(window_name, args.window_size)
    
    # Process all images
    for input_path in find_images(args.input_dir):
        try:
            I = cv2.imread(input_path)
            
            # Use advanced blur detection with center-first approach and feature density check
            result = advanced_blur_detect(I, args.threshold, args.MinZero, 
                                        args.conservative_threshold, args.feature_sensitivity)
            
            # Extract values for compatibility and output
            per = result['per']
            blurext = result['blur_extent']
            classification = result['classification']
            quality_score = result['quality_score']
            processing_info = result['processing_info']
            
            # Store results
            results.append({
                "input_path": input_path, 
                "quality_score": quality_score,  # Primary metric (0-100)
                "blur_extent": blurext,  # Primary blur metric (0-1, higher = more blur)
                "per": per,  # Edge structure metric (0-1, higher = sharper edges)
                "is_blur": classification,  # Enhanced classification considering feature density
                "center_quality": result['center_quality'],
                "processing_method": processing_info,
                "is_low_feature": result.get('is_low_feature', False),  # Feature density analysis
                "feature_metrics": result.get('feature_metrics', {})  # Detailed feature analysis
            })
            
            # Check for false cases if enabled
            if args.show_false_cases and expected_classification is not None:
                if expected_classification != classification:
                    # Display false cases in a new window if enabled
                    false_cases.append({
                        'path': input_path,
                        'ground_truth': expected_classification,
                        'predicted': classification,
                        'quality': quality_score,
                        'blur_extent': blurext,
                        'per': per
                    })
                    if len(false_cases) < 50:
                        display_intermediate_results(I, result, input_path)

                    print(f"FALSE CASE: {input_path} - Should be {'Blur' if expected_classification else 'Sharp'}, got {'Blur' if classification else 'Sharp'}")
            
            # Enhanced console output with feature density info
            feature_status = "Low-feature" if result.get('is_low_feature', False) else "Normal-feature"
            print("{0}, Quality: {1}, BlurExtent: {2:.3f}, Per: {3:.5f}, is blur: {4}, {5}, {6}".format(
                input_path, quality_score, blurext, per, classification, processing_info, feature_status))
            
            # Display intermediate results if not disabled
            if not args.no_display:
                key = display_intermediate_results(I, result, input_path)
                if key == ord('q'):
                    print("Exiting...")
                    break
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            pass
    
    # Close all windows
    if not args.no_display:
        cv2.destroyAllWindows()
    
    # Display false cases summary if enabled
    if args.show_false_cases:
        total_images = len(results)
        false_count = len(false_cases)
        accuracy = ((total_images - false_count) / total_images * 100) if total_images > 0 else 0
        
        print(f"\n📊 Classification Summary:")
        print(f"Total images processed: {total_images}")
        print(f"Correctly classified: {total_images - false_count}")
        print(f"False classifications: {false_count}")
        print(f"Accuracy: {accuracy:.1f}%")
        
    # Save results if requested
    if args.save_path:
        assert os.path.splitext(args.save_path)[1] == ".json", "You must include the extension .json on the end of the save path"
        
        with open(args.save_path, 'w') as outfile:
            json.dump(results, outfile, sort_keys=True, indent=4)
            outfile.write("\n")
        
        print(f"📁 Results saved to: {args.save_path}")
        

if __name__ == '__main__':                
    main() 