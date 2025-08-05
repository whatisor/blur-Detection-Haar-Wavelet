#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blur Detection Algorithms using Haar Wavelet Transform

Contains core blur detection algorithms and feature analysis functions.
"""

import pywt
import cv2
import numpy as np


def extract_center_region(img, center_ratio=0.5):
    """Extract center region of image with proper size constraints"""
    h, w = img.shape[:2]
    center_h = int(h * center_ratio)
    center_w = int(w * center_ratio)
    
    # Ensure center region is divisible by 16 for wavelet processing
    center_h = int(center_h / 16) * 16
    center_w = int(center_w / 16) * 16
    
    # Ensure minimum size
    center_h = max(center_h, 32)  # Minimum 32 pixels
    center_w = max(center_w, 32)
    
    start_y = (h - center_h) // 2
    start_x = (w - center_w) // 2
    
    return img[start_y:start_y + center_h, start_x:start_x + center_w]


def calculate_feature_density(img, threshold):
    """Calculate feature/edge density to distinguish low-feature images from blurred images"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Calculate various feature metrics
    h, w = gray.shape
    
    # 1. Edge density using Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    # 2. Local variance (texture measure)
    kernel = np.ones((5, 5), np.float32) / 25
    mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    variance_img = cv2.filter2D((gray.astype(np.float32) - mean_img) ** 2, -1, kernel)
    avg_variance = np.mean(variance_img)
    
    # 3. Gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = np.mean(gradient_magnitude)
    
    # 4. High frequency content (using simple high-pass filter)
    kernel_highpass = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    high_freq = cv2.filter2D(gray.astype(np.float32), -1, kernel_highpass)
    high_freq_energy = np.mean(np.abs(high_freq))
    
    return {
        'edge_density': edge_density,
        'avg_variance': avg_variance,
        'avg_gradient': avg_gradient,
        'high_freq_energy': high_freq_energy
    }


def is_low_feature_image(feature_metrics, sensitivity=0.75):
    """Determine if image has inherently low features (not blurred)"""
    # Thresholds for low-feature detection (may need tuning)
    low_feature_thresholds = {
        'edge_density': 0.02,      # Very few edges
        'avg_variance': 100,       # Low texture variance
        'avg_gradient': 15,        # Low gradient magnitude
        'high_freq_energy': 20     # Low high-frequency content
    }
    
    # Count how many metrics indicate low features
    low_feature_count = 0
    total_metrics = len(low_feature_thresholds)
    
    for metric, threshold in low_feature_thresholds.items():
        if feature_metrics[metric] < threshold:
            low_feature_count += 1
    
    # If most metrics indicate low features, it's likely a simple image, not blurred
    return low_feature_count >= (total_metrics * sensitivity)


def is_dark_image(img, dark_threshold=50):
    """Detect if image is dark based on brightness statistics"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Calculate brightness metrics
    mean_brightness = np.mean(gray)
    median_brightness = np.median(gray)
    
    # Image is considered dark if both mean and median are below threshold
    return mean_brightness < dark_threshold and median_brightness < dark_threshold


def enhance_dark_image(img, method='clahe'):
    """
    Enhance dark images for better blur detection
    
    Args:
        img: Input grayscale image
        method: Enhancement method ('clahe', 'gamma', 'combined')
        
    Returns:
        Enhanced grayscale image
    """
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img)
    
    elif method == 'gamma':
        # Gamma correction for dark images
        gamma = 0.5  # Makes dark areas brighter
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    elif method == 'combined':
        # Apply both CLAHE and mild gamma correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Mild gamma correction
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(enhanced, table)
    
    else:
        return img


def enhance_dark_image_color(img, method='combined'):
    """
    Enhance dark color images for preview purposes
    
    Args:
        img: Input BGR color image
        method: Enhancement method ('clahe', 'gamma', 'combined')
        
    Returns:
        Enhanced BGR color image
    """
    if method == 'clahe':
        # Apply CLAHE to each channel in LAB color space for better results
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply to L channel only
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif method == 'gamma':
        # Gamma correction for dark images
        gamma = 0.5  # Makes dark areas brighter
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    elif method == 'combined':
        # Apply CLAHE in LAB space then gamma correction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])  # Apply to L channel only
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Mild gamma correction
        gamma = 0.7
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(enhanced, table)
    
    else:
        return img


def calculate_adaptive_threshold(edge_maps, base_threshold, is_dark=False):
    """
    Calculate adaptive thresholds based on edge magnitude distribution
    
    Args:
        edge_maps: Tuple of (E1, E2, E3) edge magnitude maps
        base_threshold: Original threshold value
        is_dark: Whether the image is detected as dark
        
    Returns:
        Tuple of adaptive thresholds for each scale
    """
    E1, E2, E3 = edge_maps
    
    if is_dark:
        # For dark images, use percentile-based thresholds
        # This adapts to the actual distribution of edge magnitudes
        thresh1 = max(np.percentile(E1.flatten(), 85), base_threshold * 0.3)
        thresh2 = max(np.percentile(E2.flatten(), 85), base_threshold * 0.3) 
        thresh3 = max(np.percentile(E3.flatten(), 85), base_threshold * 0.3)
    else:
        # For normal images, use the original thresholds
        thresh1 = thresh2 = thresh3 = base_threshold
    
    return thresh1, thresh2, thresh3


def blur_detect(img, threshold, dark_threshold=50, enable_dark_enhancement=True):
    """
    Core Haar wavelet blur detection algorithm with dark image enhancement
    
    Args:
        img: Input image (BGR format)
        threshold: Edge detection threshold
        dark_threshold: Brightness threshold to detect dark images (0-255)
        enable_dark_enhancement: Whether to apply enhancement for dark images
        
    Returns:
        tuple: (Per, BlurExtent, E1, E2, E3, is_dark_processed, enhanced_image)
            - Per: Percentage of sharp edge structures
            - BlurExtent: Blur extent ratio (0-1, higher = more blur)
            - E1, E2, E3: Edge maps at different scales
            - is_dark_processed: Whether dark image enhancement was applied
            - enhanced_image: Enhanced grayscale image (if enhancement applied), None otherwise
    """
    # Convert image to grayscale
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_Y = Y.copy()  # Keep original for comparison
    
    # Detect if image is dark and apply enhancement if needed
    is_dark = is_dark_image(Y, dark_threshold)
    is_dark_processed = False
    enhanced_image = None
    
    if is_dark and enable_dark_enhancement:
        Y = enhance_dark_image(Y, method='combined')  # Use combined enhancement for best results
        enhanced_image = Y.copy()  # Store enhanced image for preview
        is_dark_processed = True
    
    M, N = Y.shape
    
    # Crop input image to be divisible by 16
    Y = Y[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1: Compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(Y, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3 - Use adaptive thresholds for dark images
    thresh1, thresh2, thresh3 = calculate_adaptive_threshold((E1, E2, E3), threshold, is_dark)
    EdgePoint1 = Emax1 > thresh1
    EdgePoint2 = Emax2 > thresh2
    EdgePoint3 = Emax3 > thresh3
    
    # Rule 1 Edge Points
    EdgePoint = EdgePoint1 | EdgePoint2 | EdgePoint3  # Creates boolean True/False
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint])
    
    # Rule 3 Roof-Structure or Gstep-Structure
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 
    BlurC = np.zeros((n_edges))

    for i in range(n_edges):
        if RGstructure[i] == 1 or RSstructure[i] == 1:
            if Emax1[i] < thresh1:  # Use adaptive threshold for consistency
                BlurC[i] = 1                        
        
    # Step 6
    edge_point_sum = np.sum(EdgePoint)
    if edge_point_sum == 0:
        Per = 0.0  # No edges detected, assume most blurred
    else:
        Per = np.sum(DAstructure) / edge_point_sum
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        BlurExtent = 1.0  # Fixed: should be 1.0 (most blur), not 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, BlurExtent, E1, E2, E3, is_dark_processed, enhanced_image


def advanced_blur_detect(img, threshold, min_zero_threshold, conservative_threshold=0.7, feature_sensitivity=0.75, 
                        dark_threshold=50, enable_dark_enhancement=False):
    """
    Advanced blur detection with center-first approach, feature density check, and dark image enhancement
    
    Args:
        img: Input image (BGR format)
        threshold: Edge detection threshold
        min_zero_threshold: Minimum threshold for Per value
        conservative_threshold: Higher threshold for low-feature images
        feature_sensitivity: Sensitivity for low-feature detection
        dark_threshold: Brightness threshold to detect dark images (0-255)
        enable_dark_enhancement: Whether to apply enhancement for dark images
        
    Returns:
        dict: Complete analysis results including quality score, classification, and metadata
    """
    # Validate input image
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")
    
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        raise ValueError("Image too small for processing (minimum 64x64)")
    
    # Step 1: Detect blur on center region (50% width, 50% height)
    center_img = extract_center_region(img, 0.5)
    center_per, center_blurext, center_E1, center_E2, center_E3, center_dark_processed, center_enhanced = blur_detect(
        center_img, threshold, dark_threshold, enable_dark_enhancement)
    
    # Step 1.5: Analyze feature density to detect low-feature images
    feature_metrics = calculate_feature_density(center_img, threshold)
    is_low_feature = is_low_feature_image(feature_metrics, feature_sensitivity)
    
    # Convert center blur extent to quality score for decision making
    from utils import calculate_quality_score
    center_quality = calculate_quality_score(center_blurext)
    
    # Step 2: Decide whether to process full image
    # Process full image only if center blur_extent is between 0.3-0.8 as requested
    # Logic: 
    # - BlurExtent < 0.3: Center is clear, likely whole image is clear → use center result
    # - BlurExtent > 0.8: Center is very blurry, likely whole image is blurry → use center result  
    # - BlurExtent 0.3-0.8: Uncertain case, need full image analysis for accurate assessment
    # We actually always need to process full image because sometime image center is mostly non-feature.
    process_full = True #0.3 <= center_blurext <= 0.8
    
    if process_full:
        # Process full image
        full_per, full_blurext, full_E1, full_E2, full_E3, full_dark_processed, full_enhanced = blur_detect(
            img, threshold, dark_threshold, enable_dark_enhancement)
        final_per = full_per
        final_blurext = full_blurext
        final_E1, final_E2, final_E3 = full_E1, full_E2, full_E3
        dark_processed = full_dark_processed
        enhanced_image = full_enhanced
        processing_info = "Full image processed"
    else:
        # Use center results - resize edge maps to match full image scale for visualization
        final_per = center_per
        final_blurext = center_blurext
        dark_processed = center_dark_processed
        enhanced_image = center_enhanced
        
        # Scale edge maps to represent full image dimensions for consistent visualization
        scale_factor = 2  # Center is 50% of original
        final_E1 = cv2.resize(center_E1, (center_E1.shape[1] * scale_factor, center_E1.shape[0] * scale_factor))
        final_E2 = cv2.resize(center_E2, (center_E2.shape[1] * scale_factor, center_E2.shape[0] * scale_factor))  
        final_E3 = cv2.resize(center_E3, (center_E3.shape[1] * scale_factor, center_E3.shape[0] * scale_factor))
        
        processing_info = "Center-only processed"
    
    final_blurext = min(final_blurext, center_blurext)
    final_per = max(final_per, center_per)
    # Calculate final quality score and classification using BlurExtent with feature density adjustment
    quality_score = calculate_quality_score(final_blurext)
    
    # Enhanced classification logic that considers feature density
    if is_low_feature:
        # For low-feature images, be more conservative about blur classification
        # Require higher BlurExtent threshold to classify as blurred
        #classification = final_per < conservative_threshold * min_zero_threshold
        # Default require 90% confidence( quality score < 10) to classify as blurred event.
        classification = quality_score < 10 * conservative_threshold or ( quality_score < 50 * conservative_threshold and final_per < conservative_threshold * min_zero_threshold)
        processing_note = f"{processing_info} (Low-feature detected, conservative threshold {conservative_threshold} used)"
    else:
        # Standard classification for normal feature density images
        #classification = final_per < min_zero_threshold
        # Default require 90% confidence( quality score < 10) to classify as blurred event.
        classification = quality_score < 10 or ( quality_score < 50 and final_per < min_zero_threshold)
        processing_note = processing_info
    
    # Add dark image processing information
    if dark_processed:
        quality_score /= 2
        classification = quality_score < 20 or ( quality_score < 50 and final_per < min_zero_threshold)

        processing_note += " (Dark image enhancement applied)"
    
    return {
        'per': final_per,  # Percentage of sharp edge structures (0-1, higher = sharper edges)
        'blur_extent': final_blurext,  # Blur extent ratio (0-1, higher = more blur) - PRIMARY METRIC
        'quality_score': quality_score,  # Derived from blur_extent (0-100, higher = better quality)
        'classification': classification,  # Enhanced classification considering feature density
        'edge_maps': (final_E1, final_E2, final_E3),
        'center_quality': center_quality,
        'processing_info': processing_note,  # Updated to include feature density and dark image info
        'processed_full': process_full,
        'is_low_feature': is_low_feature,  # Whether image has low feature density
        'feature_metrics': feature_metrics,  # Detailed feature analysis
        'is_dark_processed': dark_processed,  # Whether dark image enhancement was applied
        'enhanced_image': enhanced_image  # Enhanced grayscale image (if enhancement applied), None otherwise
    } 