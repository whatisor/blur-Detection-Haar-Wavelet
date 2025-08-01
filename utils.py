#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for Blur Detection

Contains display functions, file handling utilities, and helper functions.
"""

import cv2
import numpy as np
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def calculate_quality_score(blur_extent):
    """Convert BlurExtent to quality score (0-100)"""
    # BlurExtent ranges from 0 to 1, where 1 = most blur, 0 = no blur
    # Convert to 0-100 scale where 0 = most blurred, 100 = best quality
    if blur_extent >= 1:
        return 0  # Most blurred
    elif blur_extent <= 0:
        return 100  # Best quality
    else:
        # Invert the scale: quality = (1 - blur_extent) * 100
        quality = (1 - blur_extent) * 100
        return int(quality)


def get_quality_color(quality_score):
    """Get BGR color for quality score (red=0, yellow=50, green=100)"""
    # Create smooth transition: red (0) → yellow (50) → green (100)
    # BGR format: (Blue, Green, Red)
    ratio = quality_score / 100.0
    
    if ratio <= 0.5:
        # Red to Yellow transition
        # Red: (0, 0, 255) → Yellow: (0, 255, 255)
        transition = ratio * 2  # 0 to 1
        blue = 0
        green = int(255 * transition)  # 0 to 255
        red = 255  # stays at max
    else:
        # Yellow to Green transition  
        # Yellow: (0, 255, 255) → Green: (0, 255, 0)
        transition = (ratio - 0.5) * 2  # 0 to 1
        blue = 0
        green = 255  # stays at max
        red = int(255 * (1 - transition))  # 255 to 0
    
    return (blue, green, red)


def display_intermediate_results(img, result_dict, input_path):
    """
    Display the original image and intermediate edge maps with quality score
    Uses fixed window dimensions and consistent text sizing regardless of input image size
    """
    # Extract values from result dictionary
    E1, E2, E3 = result_dict['edge_maps']
    per = result_dict['per']
    blurext = result_dict['blur_extent']
    classification = result_dict['classification']
    quality_score = result_dict['quality_score']
    center_quality = result_dict['center_quality']
    processing_info = result_dict['processing_info']
    processed_full = result_dict['processed_full']
    
    # Fixed display dimensions - independent of input image size
    FIXED_MAIN_WIDTH = 800   # Fixed main image display width
    FIXED_MAIN_HEIGHT = 600  # Fixed main image display height
    FIXED_INFO_HEIGHT = 220  # Fixed info panel height
    FIXED_EDGE_WIDTH = 120   # Fixed edge map width
    FIXED_EDGE_HEIGHT = 90   # Fixed edge map height
    
    # Get original image dimensions
    original_height, original_width = img.shape[:2]
    
    # Calculate aspect ratio preserving dimensions for main image
    img_aspect = original_width / original_height
    display_aspect = FIXED_MAIN_WIDTH / FIXED_MAIN_HEIGHT
    
    if img_aspect > display_aspect:
        # Image is wider - fit by width
        display_width = FIXED_MAIN_WIDTH
        display_height = int(FIXED_MAIN_WIDTH / img_aspect)
        # Center vertically in fixed area
        y_offset = (FIXED_MAIN_HEIGHT - display_height) // 2
        x_offset = 0
    else:
        # Image is taller - fit by height  
        display_height = FIXED_MAIN_HEIGHT
        display_width = int(FIXED_MAIN_HEIGHT * img_aspect)
        # Center horizontally in fixed area
        x_offset = (FIXED_MAIN_WIDTH - display_width) // 2
        y_offset = 0
    
    # Create fixed-size main image area with black background
    main_img_area = np.zeros((FIXED_MAIN_HEIGHT, FIXED_MAIN_WIDTH, 3), dtype=np.uint8)
    
    # Resize input image to calculated display size (preserving aspect ratio)
    img_resized = cv2.resize(img, (display_width, display_height))
    
    # Place resized image in the center of fixed area
    main_img_area[y_offset:y_offset+display_height, x_offset:x_offset+display_width] = img_resized
    
    # Draw center region overlay on the resized image area only
    center_w = display_width // 2
    center_h = display_height // 2
    center_start_x = x_offset + (display_width - center_w) // 2
    center_start_y = y_offset + (display_height - center_h) // 2
    cv2.rectangle(main_img_area, (center_start_x, center_start_y), 
                  (center_start_x + center_w, center_start_y + center_h), (255, 255, 0), 3)
    
    # Fixed text scaling - independent of image size
    base_font_scale = 0.6  # Consistent base font scale
    quality_font_scale = 2.0  # Consistent quality score font scale
    
    # Draw large quality score on main image with fixed positioning
    quality_color = get_quality_color(quality_score)
    quality_text = f"Quality: {quality_score}"
    
    # Fixed positioning for quality score (top-right of actual image area)
    text_thickness = max(3, int(quality_font_scale * 2))
    outline_thickness = text_thickness + 3
    (text_width, text_height), baseline = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, quality_font_scale, text_thickness)
    
    quality_x = x_offset + display_width - text_width - 20
    quality_y = y_offset + text_height + 20
    
    # Ensure quality text stays within the display area
    if quality_x < x_offset + 20:
        quality_x = x_offset + 20
    if quality_y > y_offset + display_height - 20:
        quality_y = y_offset + display_height - 20
    
    # Add black outline for better visibility - high quality antialiased text
    cv2.putText(main_img_area, quality_text, (quality_x, quality_y), cv2.FONT_HERSHEY_SIMPLEX, quality_font_scale, (0, 0, 0), outline_thickness)
    cv2.putText(main_img_area, quality_text, (quality_x, quality_y), cv2.FONT_HERSHEY_SIMPLEX, quality_font_scale, quality_color, text_thickness)
    
    # Create fixed-size info panel
    info_img = np.zeros((FIXED_INFO_HEIGHT, FIXED_MAIN_WIDTH, 3), dtype=np.uint8)
    filename = os.path.basename(input_path)
    
    # Fixed text layout - consistent across all images
    num_lines = 5
    available_text_height = FIXED_INFO_HEIGHT - 40
    line_height = available_text_height // num_lines  # Fixed line height ~36px
    y_start = 30
    text_thickness = 1
    
    # Line 1: Filename - high quality antialiased text
    cv2.putText(info_img, filename, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, (255, 255, 255), text_thickness)
    
    # Line 2: Processing method and blur classification - high quality antialiased text
    line2_text = f"{processing_info} - Is Blur: {classification}"
    cv2.putText(info_img, line2_text, (20, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, (0, 255, 0) if not classification else (0, 0, 255), text_thickness)
    
    # Line 3: Quality scores with individual colors
    center_quality_color = get_quality_color(center_quality)
    final_quality_color = get_quality_color(quality_score)
    
    # Draw center quality - high quality antialiased text
    center_text = f"Center Quality: {center_quality}"
    cv2.putText(info_img, center_text, (20, y_start + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, center_quality_color, text_thickness)
    
    # Get center text width to position final quality text
    (center_text_width, _), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, text_thickness)
    
    # Draw separator and final quality - high quality antialiased text
    separator_text = " | Final Quality: "
    final_text = str(quality_score)
    separator_x = 20 + center_text_width
    final_x = separator_x + cv2.getTextSize(separator_text, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, text_thickness)[0][0]
    
    cv2.putText(info_img, separator_text, (separator_x, y_start + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, (200, 200, 200), text_thickness)
    cv2.putText(info_img, final_text, (final_x, y_start + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.85, final_quality_color, text_thickness)
    
    # Line 4: Technical details with feature density info - high quality antialiased text
    is_low_feature = result_dict.get('is_low_feature', False)
    feature_status = "Low-feature" if is_low_feature else "Normal-feature"
    line4_text = f"BlurExtent: {blurext:.3f} | Per: {per:.5f} | {feature_status}"
    feature_color = (100, 200, 255) if is_low_feature else (200, 200, 200)  # Light blue for low-feature
    cv2.putText(info_img, line4_text, (20, y_start + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.75, feature_color, text_thickness)
    
    # Line 5: Instructions - high quality antialiased text
    line5_text = "Press any key to continue, 'q' to quit | Drag edges to resize"
    cv2.putText(info_img, line5_text, (20, y_start + line_height * 4), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 0.7, (100, 255, 255), text_thickness)
    
    # Create main content area (image + info panel)
    main_content = np.vstack((main_img_area, info_img))
    
    # Process edge maps with fixed dimensions
    # Resize edge maps to fixed size regardless of original edge map dimensions
    E1_display = cv2.resize(E1, (FIXED_EDGE_WIDTH, FIXED_EDGE_HEIGHT))
    E2_display = cv2.resize(E2, (FIXED_EDGE_WIDTH, FIXED_EDGE_HEIGHT))
    E3_display = cv2.resize(E3, (FIXED_EDGE_WIDTH, FIXED_EDGE_HEIGHT))
    
    # Normalize edge maps for display
    E1_norm = cv2.normalize(E1_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    E2_norm = cv2.normalize(E2_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    E3_norm = cv2.normalize(E3_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Create fixed-size edge panel
    edge_panel_width = FIXED_EDGE_WIDTH + 40  # Fixed width with margins
    edge_panel_height = FIXED_MAIN_HEIGHT + FIXED_INFO_HEIGHT  # Match main content height
    edge_panel = np.zeros((edge_panel_height, edge_panel_width, 3), dtype=np.uint8)
    
    # Fixed spacing for edge maps
    spacing_between = 40  # Fixed spacing
    current_y = spacing_between
    
    # Convert edge maps to 3-channel
    E1_3ch = cv2.cvtColor(E1_norm, cv2.COLOR_GRAY2BGR)
    E2_3ch = cv2.cvtColor(E2_norm, cv2.COLOR_GRAY2BGR)  
    E3_3ch = cv2.cvtColor(E3_norm, cv2.COLOR_GRAY2BGR)
    
    # Fixed label font scale
    label_font_scale = 0.5
    
    # Add edge maps and labels with fixed positioning - high quality antialiased text
    # Edge Map L1
    cv2.putText(edge_panel, "Edge L1 (Fine)", (10, current_y - 5), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), 1)
    edge_panel[current_y:current_y+FIXED_EDGE_HEIGHT, 10:10+FIXED_EDGE_WIDTH] = E1_3ch
    current_y += FIXED_EDGE_HEIGHT + spacing_between
    
    # Edge Map L2
    cv2.putText(edge_panel, "Edge L2 (Medium)", (10, current_y - 5), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), 1)
    edge_panel[current_y:current_y+FIXED_EDGE_HEIGHT, 10:10+FIXED_EDGE_WIDTH] = E2_3ch
    current_y += FIXED_EDGE_HEIGHT + spacing_between
    
    # Edge Map L3
    cv2.putText(edge_panel, "Edge L3 (Coarse)", (10, current_y - 5), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), 1)
    edge_panel[current_y:current_y+FIXED_EDGE_HEIGHT, 10:10+FIXED_EDGE_WIDTH] = E3_3ch
    
    # Combine main content with edge panel (main content on left, edge maps on right)
    combined_display = np.hstack((main_content, edge_panel))
    
    # Display the combined image
    cv2.imshow('Advanced Blur Detection - Main Focus | Edge Sidebar', combined_display)
    
    # Wait for key press
    key = cv2.waitKey(0) & 0xFF
    return key


def create_false_cases_display(false_cases):
    """Create a grid display of all false classification cases"""
    if not false_cases:
        print("No false cases found!")
        return
    
    print(f"Found {len(false_cases)} false classification cases")
    
    # Calculate grid dimensions
    num_cases = len(false_cases)
    cols = min(4, num_cases)  # Max 4 columns
    rows = (num_cases + cols - 1) // cols
    
    # Thumbnail size
    thumb_size = 200
    margin = 10
    text_height = 60
    
    # Create display image with title space
    title_bg_height = 50
    display_width = cols * (thumb_size + margin) + margin
    display_height = rows * (thumb_size + text_height + margin) + margin + title_bg_height
    false_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    for idx, case in enumerate(false_cases):
        row = idx // cols
        col = idx % cols
        
        # Calculate position (offset by title height)
        x = col * (thumb_size + margin) + margin
        y = row * (thumb_size + text_height + margin) + margin + title_bg_height
        
        # Load and resize image
        img = cv2.imread(case['path'])
        if img is not None:
            img_resized = cv2.resize(img, (thumb_size, thumb_size))
            false_display[y:y+thumb_size, x:x+thumb_size] = img_resized
            
            # Add text information
            filename = os.path.basename(case['path'])
            if len(filename) > 25:
                filename = filename[:22] + "..."
            
            # Text background for readability
            text_bg_y = y + thumb_size
            cv2.rectangle(false_display, (x, text_bg_y), (x + thumb_size, text_bg_y + text_height), (40, 40, 40), -1)
            
            # Add text - high quality antialiased text (50% smaller)
            font_scale = 0.3  # 50% smaller than original 0.4
            thickness = 1
            
            cv2.putText(false_display, filename, (x + 5, text_bg_y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Ground truth vs prediction - high quality antialiased text
            truth_text = f"Should: {'Blur' if case['ground_truth'] else 'Sharp'}"
            pred_text = f"Got: {'Blur' if case['predicted'] else 'Sharp'}"
            quality_text = f"Quality: {case['quality']}"
            
            cv2.putText(false_display, truth_text, (x + 5, text_bg_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 255, 100), thickness)
            cv2.putText(false_display, pred_text, (x + 5, text_bg_y + 45), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 255), thickness)
            cv2.putText(false_display, quality_text, (x + 5, text_bg_y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 100), thickness)
    
    # Add title with padding at top
    cv2.rectangle(false_display, (0, 0), (display_width, title_bg_height), (20, 20, 20), -1)
    
    title_text = f"False Classification Cases ({len(false_cases)} found) - Press any key to close | Resizable"
    title_y = 30
    cv2.putText(false_display, title_text, (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 50% smaller with antialiasing
    
    # Display window with better resizing properties
    false_window_name = 'False Classification Cases'
    cv2.namedWindow(false_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Set initial window size based on content, but allow user resizing
    initial_width = min(1400, display_width)
    initial_height = min(900, display_height)
    cv2.resizeWindow(false_window_name, initial_width, initial_height)
    cv2.imshow(false_window_name, false_display)
    
    print("False cases window opened. Press any key to close...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(false_window_name)

    if key == ord('q'):
        return True
    else:
        return False


def find_images(input_dir):
    """Find all image files in the input directory"""
    extensions = [".jpg", ".png", ".jpeg"]

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                yield os.path.join(root, file)


def setup_window(window_name, window_size):
    """Setup OpenCV window with proper configuration"""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    # Parse window size from command line argument
    try:
        width, height = map(int, window_size.split('x'))
        cv2.resizeWindow(window_name, width, height)
        print(f"📺 Window initialized at {width}x{height} (resizable with mouse)")
        return True
    except ValueError:
        cv2.resizeWindow(window_name, 1200, 800)
        print(f"📺 Invalid window size format '{window_size}', using default 1200x800")
        return False 


def visualize_quality_distribution(results, save_path=None, show_plot=True):
    """
    Visualize quality score distribution with comprehensive analysis
    
    Args:
        results: List of result dictionaries containing quality scores and classifications
        save_path: Optional path to save the plot (e.g., 'quality_distribution.png')
        show_plot: Whether to display the plot interactively
        
    Returns:
        dict: Statistics about the quality distribution
    """
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Error: matplotlib not available. Install with: pip install matplotlib")
        return None
    
    if not results:
        print("❌ Error: No results to visualize")
        return None
    
    # Extract data
    quality_scores = [r['quality_score'] for r in results]
    blur_classifications = [r['is_blur'] for r in results]
    dark_processed = [r.get('is_dark_processed', False) for r in results]
    low_feature = [r.get('is_low_feature', False) for r in results]
    blur_extents = [r['blur_extent'] for r in results]
    pers = [r['per'] for r in results]
    
    # Separate scores by classification
    sharp_scores = [q for q, is_blur in zip(quality_scores, blur_classifications) if not is_blur]
    blur_scores = [q for q, is_blur in zip(quality_scores, blur_classifications) if is_blur]
    sharp_blur_extents = [b for b, is_blur in zip(blur_extents, blur_classifications) if not is_blur]
    blur_blur_extents = [b for b, is_blur in zip(blur_extents, blur_classifications) if is_blur]
    sharp_pers = [p for p, is_blur in zip(pers, blur_classifications) if not is_blur]
    blur_pers = [p for p, is_blur in zip(pers, blur_classifications) if is_blur]
    
    # Separate by processing type
    dark_enhanced_scores = [q for q, is_dark in zip(quality_scores, dark_processed) if is_dark]
    normal_brightness_scores = [q for q, is_dark in zip(quality_scores, dark_processed) if not is_dark]
    
    # Calculate statistics
    stats = {
        'total_images': len(results),
        'overall_mean': np.mean(quality_scores),
        'overall_median': np.median(quality_scores),
        'overall_std': np.std(quality_scores),
        'sharp_images': len(sharp_scores),
        'blur_images': len(blur_scores),
        'sharp_mean': np.mean(sharp_scores) if sharp_scores else 0,
        'blur_mean': np.mean(blur_scores) if blur_scores else 0,
        'dark_enhanced_count': len(dark_enhanced_scores),
        'dark_enhanced_mean': np.mean(dark_enhanced_scores) if dark_enhanced_scores else 0,
        'low_feature_count': sum(low_feature)
    }
    
    # Create the visualization
    fig, axs = plt.subplots(3, 2, figsize=(18, 16))
    fig.suptitle('Quality Score, Blur Extent, and Per Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall quality distribution
    ax1 = axs[0, 0]
    ax1.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(stats['overall_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["overall_mean"]:.1f}')
    ax1.axvline(stats['overall_median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {stats["overall_median"]:.1f}')
    ax1.set_xlabel('Quality Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Overall Quality Distribution (n={stats["total_images"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharp vs Blur quality comparison
    ax2 = axs[0, 1]
    bins = np.linspace(0, 100, 21)
    ax2.hist([sharp_scores, blur_scores], bins=bins, alpha=0.7, 
             color=['green', 'red'], label=['Sharp', 'Blur'], edgecolor='black')
    ax2.set_xlabel('Quality Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Sharp vs Blur Quality\nSharp: {len(sharp_scores)}, Blur: {len(blur_scores)}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Blur Extent distribution (sharp vs blur)
    ax3 = axs[1, 0]
    bins_blur = np.linspace(0, 1, 21)
    ax3.hist([sharp_blur_extents, blur_blur_extents], bins=bins_blur, alpha=0.7, 
             color=['green', 'red'], label=['Sharp', 'Blur'], edgecolor='black')
    ax3.set_xlabel('Blur Extent')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Blur Extent Distribution (Sharp vs Blur)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Per distribution (sharp vs blur)
    ax4 = axs[1, 1]
    bins_per = np.linspace(0, 1, 21)
    ax4.hist([sharp_pers, blur_pers], bins=bins_per, alpha=0.7, 
             color=['green', 'red'], label=['Sharp', 'Blur'], edgecolor='black')
    ax4.set_xlabel('Per (Edge Structure)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Per Distribution (Sharp vs Blur)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot: Blur Extent vs Per, colored by classification
    ax5 = axs[2, 0]
    scatter = ax5.scatter(blur_extents, pers, c=blur_classifications, cmap='coolwarm', alpha=0.7, edgecolor='k')
    ax5.set_xlabel('Blur Extent')
    ax5.set_ylabel('Per (Edge Structure)')
    ax5.set_title('Blur Extent vs Per (Color: Blur=1, Sharp=0)')
    legend1 = ax5.legend(*scatter.legend_elements(), title="is_blur", loc="upper right")
    ax5.add_artist(legend1)
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics summary
    ax6 = axs[2, 1]
    ax6.axis('off')
    stats_text = f"""
    📊 QUALITY STATISTICS
    
    Total Images: {stats['total_images']}
    Overall Mean: {stats['overall_mean']:.1f} ± {stats['overall_std']:.1f}
    Overall Median: {stats['overall_median']:.1f}
    
    📈 CLASSIFICATION BREAKDOWN
    Sharp Images: {stats['sharp_images']} (Mean: {stats['sharp_mean']:.1f})
    Blur Images: {stats['blur_images']} (Mean: {stats['blur_mean']:.1f})
    
    🌙 PROCESSING BREAKDOWN  
    Dark Enhanced: {stats['dark_enhanced_count']} (Mean: {stats['dark_enhanced_mean']:.1f})
    Normal Brightness: {len(normal_brightness_scores)}
    Low Feature Images: {stats['low_feature_count']}
    
    📋 QUALITY RANGES
    Excellent (90-100): {sum(1 for q in quality_scores if q >= 90)}
    Good (70-89): {sum(1 for q in quality_scores if 70 <= q < 90)}
    Fair (50-69): {sum(1 for q in quality_scores if 50 <= q < 70)}
    Poor (0-49): {sum(1 for q in quality_scores if q < 50)}
    """
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Quality distribution plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return stats 