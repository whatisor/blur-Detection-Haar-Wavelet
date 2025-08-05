#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Camera Simultaneous Blur Detection Dashboard

Shows ALL camera feeds simultaneously side-by-side with independent blur detection.
No switching - all cameras display their results in real-time.
"""

import cv2
import numpy as np
import time
import sys
import urllib.request
import urllib.error
import argparse
import threading
from queue import Queue
from detector import advanced_blur_detect, enhance_dark_image_color, is_dark_image
from utils import setup_window, get_quality_color

# Global variables for slider controls
dark_enhancement_threshold = 0  # Range: 0 to 100, 0 = disabled


def map_slider_to_threshold(slider_value):
    """
    Map slider value (0 to 100) to actual dark threshold (0-255)
    
    Args:
        slider_value: Slider value from 0 to 100, where 0 means disabled
        
    Returns:
        tuple: (enable_dark_enhancement, dark_threshold)
    """
    if slider_value == 0:
        return False, 50  # Disabled
    else:
        # Values 1-100 map to thresholds 200-10 (higher slider = more sensitive)
        # Lower threshold means more images are detected as dark
        threshold = 255 * slider_value / 100
        return True, int(threshold)


def on_dark_threshold_change(val):
    """Callback for dark enhancement threshold slider"""
    global dark_enhancement_threshold
    dark_enhancement_threshold = val  # Direct mapping from 0-100


def draw_text_sharp(img, text, pos, font, scale, color, thickness, outline_color=(0, 0, 0), outline_thickness=None):
    """Draw sharp, crisp text without antialiasing blur"""
    if outline_thickness is None:
        outline_thickness = thickness + 1
    
    # Draw outline for contrast (no antialiasing)
    cv2.putText(img, text, pos, font, scale, outline_color, outline_thickness)
    # Draw main text (no antialiasing for crisp edges)
    cv2.putText(img, text, pos, font, scale, color, thickness)


def draw_text_extra_sharp(img, text, pos, font, scale, color, thickness):
    """Draw extra sharp text with multiple passes for better edge definition"""
    # Multiple thin passes for sharper, more defined edges
    for offset in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        cv2.putText(img, text, (pos[0] + offset[0], pos[1] + offset[1]), 
                   font, scale, (0, 0, 0), thickness)
    # Main text on top
    cv2.putText(img, text, pos, font, scale, color, thickness)


def resize_with_aspect_ratio(frame, target_size, bg_color=(0, 0, 0)):
    """
    Resize frame while maintaining aspect ratio, adding letterboxing if needed
    
    Args:
        frame: Input frame to resize
        target_size: (width, height) target size
        bg_color: Background color for letterboxing (default black)
        
    Returns:
        Resized frame with maintained aspect ratio
    """
    if frame is None:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    target_width, target_height = target_size
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate aspect ratios
    frame_aspect = frame_width / frame_height
    target_aspect = target_width / target_height
    
    # Determine scaling to fit within target size while maintaining aspect ratio
    if frame_aspect > target_aspect:
        # Frame is wider - fit by width
        new_width = target_width
        new_height = int(target_width / frame_aspect)
    else:
        # Frame is taller - fit by height
        new_height = target_height
        new_width = int(target_height * frame_aspect)
    
    # Resize frame maintaining aspect ratio
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create target-sized canvas with background color
    canvas = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    
    # Calculate position to center the resized frame
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Place resized frame on canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
    
    return canvas


class CameraFeed:
    """Individual camera feed with independent blur detection"""
    
    def __init__(self, camera_id, ip, port):
        self.camera_id = camera_id
        self.ip = ip
        self.port = port
        self.url = f"http://{ip}:{port}/?action=stream"
        self.cap = None
        self.use_manual = False
        self.running = False
        self.thread = None
        
        # Latest data
        self.latest_frame = None
        self.latest_result = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.connection_status = "Connecting..."
        
    def test_connection(self):
        """Test camera connection"""
        print(f"🔍 Testing Camera {self.camera_id}: {self.ip}:{self.port}")
        
        # Try OpenCV VideoCapture first
        try:
            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if ret and frame is not None:
                self.cap = cap
                self.use_manual = False
                self.connection_status = "OpenCV Connected"
                print(f"   ✅ OpenCV - Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return True
            cap.release()
        except Exception as e:
            print(f"   ❌ OpenCV failed: {e}")
        
        # Try manual capture
        try:
            frame = self._capture_manual_frame()
            if frame is not None:
                self.use_manual = True
                self.connection_status = "Manual Connected"
                print(f"   ✅ Manual - Resolution: {frame.shape[1]}x{frame.shape[0]}")
                return True
        except Exception as e:
            print(f"   ❌ Manual failed: {e}")
        
        self.connection_status = "Failed"
        print(f"   ❌ Connection failed")
        return False
    
    def _capture_manual_frame(self):
        """Capture single frame manually"""
        try:
            with urllib.request.urlopen(self.url, timeout=3) as response:
                image_data = response.read()
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except:
            return None
    
    def _get_latest_frame(self):
        """Get the most recent frame"""
        if self.use_manual:
            return self._capture_manual_frame()
        else:
            if self.cap is None:
                return None
            
            # Flush buffer to get latest frame
            latest_frame = None
            for _ in range(5):  # Flush up to 5 frames
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    latest_frame = frame
                else:
                    break
            return latest_frame
    
    def start_processing(self, blur_params):
        """Start independent processing thread"""
        self.blur_params = blur_params
        self.running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        print(f"📷 Camera {self.camera_id} processing started")
    
    def _processing_loop(self):
        """Main processing loop for this camera"""
        fps_frame_count = 0
        fps_start_time = time.time()
        
        while self.running:
            try:
                # Capture frame
                frame = self._get_latest_frame()
                if frame is None:
                    self.connection_status = "No Frame"
                    time.sleep(0.1)
                    continue
                
                self.connection_status = "Active"
                self.frame_count += 1
                fps_frame_count += 1
                
                # Calculate FPS every second
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    self.fps = fps_frame_count / (current_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = current_time
                
                # Use dynamic slider values for dark enhancement
                global dark_enhancement_threshold
                enable_dark, dark_thresh = map_slider_to_threshold(dark_enhancement_threshold)
                
                # Run blur detection
                result = advanced_blur_detect(
                    frame,
                    self.blur_params['threshold'],
                    self.blur_params['min_zero_threshold'], 
                    self.blur_params['conservative_threshold'],
                    self.blur_params['feature_sensitivity'],
                    dark_thresh,
                    enable_dark
                )
                
                # Store latest data (thread-safe)
                self.latest_frame = frame.copy()
                self.latest_result = result.copy()
                
                # Print status occasionally
                if self.frame_count % 50 == 0:  # Every ~1.5 seconds
                    quality = result['quality_score']
                    classification = 'BLUR' if result['classification'] else 'SHARP'
                    print(f"Camera {self.camera_id} ({self.ip}:{self.port}): {classification} - Quality {quality} - {self.fps:.1f} FPS")
                
                time.sleep(0.033)  # ~30 FPS max
                
            except Exception as e:
                print(f"❌ Camera {self.camera_id} error: {e}")
                self.connection_status = f"Error: {str(e)[:20]}"
                time.sleep(1)
    
    def get_display_frame(self, display_size=(1280, 720)):
        """Get frame formatted for display with improved text and enhanced image preview"""
        # Increased info panel height for better text display
        info_height = 160
        
        if self.latest_frame is None or self.latest_result is None:
            # Create "no signal" display with better text
            total_height = display_size[1] + info_height
            display = np.zeros((total_height, display_size[0], 3), dtype=np.uint8)
            display[:] = (40, 40, 40)
            
            # Camera info with sharp, crisp text
            draw_text_extra_sharp(display, f"Camera {self.camera_id}", (15, 40), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            draw_text_extra_sharp(display, f"{self.ip}:{self.port}", (15, 75), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            draw_text_extra_sharp(display, self.connection_status, (15, 110), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
            draw_text_extra_sharp(display, "NO SIGNAL", (15, display_size[1]//2 + 70), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 3)
            return display
        
        # Check if dark enhancement is enabled and if image is dark
        global dark_enhancement_threshold
        enable_dark, dark_thresh = map_slider_to_threshold(dark_enhancement_threshold)
        result = self.latest_result
        is_dark_processed = result.get('is_dark_processed', False)
        
        if enable_dark and dark_enhancement_threshold > 0:
            # Check if current frame is dark for preview enhancement
            gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
            frame_is_dark = is_dark_image(gray, dark_thresh)
            
            if frame_is_dark:
                # Show vertical stack comparison: original on top, color-enhanced on bottom
                # Apply color enhancement for preview (much better visual feedback)
                enhanced_color = enhance_dark_image_color(self.latest_frame, method='combined')
                
                # Resize both images to fit half the display height
                half_height = display_size[1] // 2
                original_resized = resize_with_aspect_ratio(self.latest_frame, 
                                                          (display_size[0], half_height), bg_color=(20, 20, 20))
                enhanced_resized = resize_with_aspect_ratio(enhanced_color, 
                                                          (display_size[0], half_height), bg_color=(20, 20, 20))
                
                # Combine vertically (original on top, enhanced on bottom)
                frame_combined = np.vstack([original_resized, enhanced_resized])
                
                # Ensure the combined frame has exactly the expected dimensions to avoid dimension mismatch
                if frame_combined.shape[:2] != (display_size[1], display_size[0]):
                    frame_resized = cv2.resize(frame_combined, display_size)
                else:
                    frame_resized = frame_combined
                
                # Add labels to distinguish original vs enhanced
                draw_text_sharp(frame_resized, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
                draw_text_sharp(frame_resized, "Enhanced (Color)", (10, half_height + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
            else:
                # Not dark enough for enhancement, show normal frame
                frame_resized = resize_with_aspect_ratio(self.latest_frame, display_size, bg_color=(20, 20, 20))
        else:
            # Dark enhancement disabled, show normal frame
            frame_resized = resize_with_aspect_ratio(self.latest_frame, display_size, bg_color=(20, 20, 20))
        
        # Create larger info panel with gradient background for better text visibility
        info_panel = np.zeros((info_height, display_size[0], 3), dtype=np.uint8)
        
        # Create gradient background for better text contrast
        for i in range(info_height):
            gradient_value = int(60 - (i / info_height) * 20)  # 60 to 40
            info_panel[i, :] = (gradient_value, gradient_value, gradient_value)
        
        # Extract results
        result = self.latest_result
        quality_score = result['quality_score']
        blur_extent = result['blur_extent']
        per = result['per']
        classification = result['classification']
        is_low_feature = result.get('is_low_feature', False)
        is_dark_processed = result.get('is_dark_processed', False)
        
        # Get colors
        quality_color = get_quality_color(quality_score)
        blur_color = (0, 100, 255) if classification else (0, 255, 100)  # Brighter colors
        
        # Improved text layout with better spacing and sizes
        font_main = cv2.FONT_HERSHEY_SIMPLEX
        font_detail = cv2.FONT_HERSHEY_SIMPLEX
        
        # Line 1: Camera ID and IP (sharp, crisp text)
        draw_text_extra_sharp(info_panel, f"Camera {self.camera_id}", (10, 25), 
                             font_main, 0.7, (255, 255, 255), 2)
        
        # Add IP on same line if space allows, otherwise next line
        ip_text = f"{self.ip}:{self.port}"
        if len(ip_text) < 15:  # Short IP, same line
            draw_text_sharp(info_panel, f" - {ip_text}", (120, 25), 
                           font_detail, 0.5, (200, 200, 200), 1)
            next_y = 50
        else:  # Long IP, next line
            draw_text_sharp(info_panel, ip_text, (10, 45), 
                           font_detail, 0.5, (200, 200, 200), 1)
            next_y = 70
        
        # Line 2: Quality score (large, colored, sharp)
        quality_text = f"Quality: {quality_score}"
        draw_text_extra_sharp(info_panel, quality_text, (10, next_y), 
                             font_main, 0.8, quality_color, 3)
        
        # Line 3: Blur result (large, colored, sharp)
        result_text = "BLUR" if classification else "SHARP"
        draw_text_extra_sharp(info_panel, f"Result: {result_text}", (10, next_y + 25), 
                             font_main, 0.7, blur_color, 2)
        
        # Line 4: Technical details (sharp, readable)
        tech_text = f"BlurExt: {blur_extent:.3f}  Per: {per:.3f}"
        draw_text_sharp(info_panel, tech_text, (10, next_y + 50), 
                       font_detail, 0.5, (220, 220, 220), 1)
        
        # Line 5: Status (FPS and flags, sharp)
        # Note: global dark_enhancement_threshold already declared at method start
        status_parts = [f"FPS: {self.fps:.1f}"]
        if is_low_feature:
            status_parts.append("Low-feature")
        if is_dark_processed:
            status_parts.append("Dark-calc")  # Used in blur calculation
        
        # Check if color enhancement is being applied for preview
        if dark_enhancement_threshold > 0:
            enable_dark, dark_thresh = map_slider_to_threshold(dark_enhancement_threshold)
            if enable_dark and self.latest_frame is not None:
                gray = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY)
                if is_dark_image(gray, dark_thresh):
                    status_parts.append("Color-enhanced")  # Used for preview
        
        status_parts.append(f"Dark: {dark_enhancement_threshold}")
        
        status_text = " | ".join(status_parts)
        draw_text_sharp(info_panel, status_text, (10, next_y + 75), 
                       font_detail, 0.45, (150, 220, 255), 1)
        
        # Add border around info panel for better separation
        cv2.rectangle(info_panel, (0, 0), (display_size[0]-1, info_height-1), (100, 100, 100), 2)
        
        # Combine frame and info
        combined = np.vstack([frame_resized, info_panel])
        return combined
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()


def calculate_optimal_camera_size(num_cameras, max_cols=3, target_screen_size=(1920, 1080)):
    """Calculate optimal camera size to fit screen while maximizing individual camera size"""
    if num_cameras <= 0:
        return (1280, 720)
    
    cols = min(num_cameras, max_cols)
    rows = (num_cameras + cols - 1) // cols
    
    # Reserve space for title bar and spacing
    title_height = 60
    spacing = 10
    info_panel_height = 160
    
    available_width = target_screen_size[0] - (spacing * (cols + 1))
    available_height = target_screen_size[1] - title_height - (spacing * (rows + 1))
    
    # Calculate camera frame area (excluding info panel)
    camera_frame_height = available_height // rows - info_panel_height
    camera_width = available_width // cols
    
    # Ensure minimum readable size
    camera_width = max(camera_width, 320)
    camera_frame_height = max(camera_frame_height, 240)
    
    # Total height includes info panel
    camera_total_height = camera_frame_height + info_panel_height
    
    return (camera_width, camera_total_height)


def create_grid_layout(camera_feeds, max_cols=3, camera_size=(1280, 720)):
    """Create grid layout of all cameras with improved spacing"""
    if not camera_feeds:
        return np.zeros((1280, 720, 3), dtype=np.uint8)
    
    num_cameras = len(camera_feeds)
    cols = min(num_cameras, max_cols)
    rows = (num_cameras + cols - 1) // cols
    
    # Get camera displays
    camera_displays = []
    for feed in camera_feeds:
        display = feed.get_display_frame(camera_size)
        camera_displays.append(display)
    
    # Calculate grid dimensions
    display_height, display_width = camera_displays[0].shape[:2]
    grid_height = rows * display_height
    grid_width = cols * display_width
    
    # Create grid with dark background
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    grid[:] = (20, 20, 20)  # Dark background
    
    # Place cameras in grid with spacing
    spacing = 2
    for i, camera_display in enumerate(camera_displays):
        row = i // cols
        col = i % cols
        
        y1 = row * display_height + spacing
        y2 = y1 + display_height - spacing
        x1 = col * display_width + spacing
        x2 = x1 + display_width - spacing
        
        if y2 <= grid_height and x2 <= grid_width:
            # Add slight border around each camera
            cv2.rectangle(grid, (x1-spacing, y1-spacing), (x2+spacing, y2+spacing), (80, 80, 80), 1)
            grid[y1:y2, x1:x2] = camera_display[:y2-y1, :x2-x1]
    
    return grid


def main():
    """Main function for simultaneous multi-camera display"""
    parser = argparse.ArgumentParser(
        description='Multi-Camera Simultaneous Blur Detection Dashboard',
        epilog='''
Examples:
  python camera.py -c 192.168.1.100:8080 192.168.1.101:8080    # Two cameras (auto-size)
  python camera.py -c 192.168.1.100:8080 192.168.1.101:8080 192.168.1.102:8080  # Three cameras (auto-size)
  python camera.py --interactive                                # Interactive setup
  python camera.py -c 192.168.1.100:8080 --camera-size 800x600 # Custom size
  python camera.py -c 192.168.1.100:8080 --screen-size 2560x1440 # 4K display auto-size
  
Features:
- All cameras display simultaneously - no switching!
- Auto-calculated sizing for full display coverage
- Camera aspect ratio preserved with letterboxing
- Sharp, crisp text rendering - no antialiasing blur
        '''
    )
    
    parser.add_argument('-c', '--cameras', nargs='+',
                       help='Camera IP:PORT pairs for simultaneous display')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive camera setup')
    parser.add_argument('--threshold', type=float, default=35,
                       help='Blur detection threshold (default: 35)')
    parser.add_argument('--cols', type=int, default=3,
                       help='Maximum columns in grid (default: 3)')
    parser.add_argument('--camera-size', default="auto",
                       help='Each camera display size (default: auto-calculated for full display) - aspect ratio preserved')
    parser.add_argument('--screen-size', default="1920x1080",
                       help='Screen resolution for auto-sizing (default: 1920x1080)')
    # Note: Dark enhancement is now controlled by the interactive slider
    args = parser.parse_args()
    
    print("🚀 Multi-Camera Simultaneous Blur Detection Dashboard")
    print("="*60)
    print("📺 Sharp, Crisp Text Display - ALL CAMERAS SIMULTANEOUSLY!")
    print("")
    
    # Parse camera size - use optimal sizing based on number of cameras
    user_specified_size = False
    if args.camera_size == "auto":
        # Will calculate optimal size after we know number of cameras
        camera_size = (1280, 720)  # Temporary default
    else:
        try:
            cam_width, cam_height = map(int, args.camera_size.split('x'))
            camera_size = (cam_width, cam_height)
            user_specified_size = True
        except:
            print(f"⚠️  Invalid camera size format '{args.camera_size}', using auto-calculation")
            camera_size = (1280, 720)  # Temporary default
    
    # Get camera configurations
    camera_configs = []
    
    if args.interactive:
        print("🎥 Interactive Setup - Enter cameras to display simultaneously:")
        while True:
            ip_port = input(f"Camera {len(camera_configs) + 1} IP:PORT (empty to finish): ").strip()
            if not ip_port:
                break
            if ':' in ip_port:
                try:
                    ip, port = ip_port.split(':')
                    camera_configs.append((ip.strip(), int(port)))
                except ValueError:
                    print("❌ Invalid format. Use IP:PORT")
    elif args.cameras:
        for camera_input in args.cameras:
            if ':' in camera_input:
                try:
                    ip, port = camera_input.split(':')
                    camera_configs.append((ip.strip(), int(port)))
                except ValueError:
                    print(f"❌ Skipping invalid: {camera_input}")
    else:
        # Default camera
        camera_configs = [("169.254.200.200", 9080)]
    
    if not camera_configs:
        print("❌ No cameras configured!")
        sys.exit(1)
    
    # Calculate optimal camera size if user didn't specify one
    if not user_specified_size:
        # Parse screen size
        try:
            screen_width, screen_height = map(int, args.screen_size.split('x'))
            screen_size = (screen_width, screen_height)
        except:
            screen_size = (1920, 1080)  # Default fallback
            print(f"⚠️  Invalid screen size format '{args.screen_size}', using 1920x1080")
        
        camera_size = calculate_optimal_camera_size(len(camera_configs), args.cols, screen_size)
        print(f"🎯 Auto-calculated optimal camera size: {camera_size[0]}x{camera_size[1]} for {len(camera_configs)} cameras on {screen_size[0]}x{screen_size[1]} display")
    
    print(f"📋 Configuration:")
    print(f"   Cameras to display simultaneously: {len(camera_configs)}")
    for i, (ip, port) in enumerate(camera_configs):
        print(f"   Camera {i+1}: {ip}:{port}")
    print(f"   Grid layout: {args.cols} columns max")
    print(f"   Each camera size: {camera_size[0]}x{camera_size[1]} (aspect ratio preserved)")
    print(f"   Text rendering: Sharp, crisp edges - no antialiasing blur")
    print("")
    
    # Initialize camera feeds
    camera_feeds = []
    for i, (ip, port) in enumerate(camera_configs):
        feed = CameraFeed(i+1, ip, port)
        camera_feeds.append(feed)
    
    # Test all connections
    print("🔍 Testing all camera connections...")
    working_feeds = []
    for feed in camera_feeds:
        if feed.test_connection():
            working_feeds.append(feed)
    
    if not working_feeds:
        print("❌ No cameras connected successfully!")
        sys.exit(1)
    
    print(f"\n✅ {len(working_feeds)} camera(s) connected and will display simultaneously")
    
    # Blur detection parameters (dark enhancement now controlled by slider)
    blur_params = {
        'threshold': args.threshold,
        'min_zero_threshold': 0.001,
        'conservative_threshold': 0.7,
        'feature_sensitivity': 0.75
        # dark_threshold and enable_dark_enhancement are now dynamic from slider
    }
    
    # Start all camera processing threads
    print("\n🎬 Starting all cameras...")
    for feed in working_feeds:
        feed.start_processing(blur_params)
    
    # Setup display with better window properties
    window_name = 'Multi-Camera Dashboard - Sharp Text'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    # Create dark enhancement threshold slider
    # Slider range: 0-100, where 0 = disabled
    cv2.createTrackbar('Dark Enhancement (0-100, 0=off)', window_name, 0, 100, on_dark_threshold_change)
    
    print(f"\n📺 Live dashboard started - showing {len(working_feeds)} cameras simultaneously")
    print("📋 Controls: 'q' to quit, 's' to save screenshot")
    print("🎛️  Dark Enhancement Slider: 0-100 (0 = disabled)")
    print("   - Higher values: More sensitive to dark images (enhances more images)") 
    print("✨ Features: Sharp crisp text, auto-sized for full display, aspect ratio preserved")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Create grid display of all cameras
            grid_display = create_grid_layout(working_feeds, args.cols, camera_size)
            
            # Add improved title bar
            title_height = 60
            title_width = grid_display.shape[1]
            title_bar = np.zeros((title_height, title_width, 3), dtype=np.uint8)
            
            # Gradient background for title
            for i in range(title_height):
                gradient_value = int(80 - (i / title_height) * 20)  # 80 to 60
                title_bar[i, :] = (gradient_value, gradient_value, gradient_value)
            
            # Sharp, crisp title text
            title_text = f"Live Multi-Camera Dashboard - {len(working_feeds)} Cameras - Frame {frame_count}"
            draw_text_extra_sharp(title_bar, title_text, (20, 35), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add timestamp with sharp rendering
            timestamp = time.strftime("%H:%M:%S")
            draw_text_sharp(title_bar, timestamp, (title_width - 120, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            
            # Combine title and grid
            final_display = np.vstack([title_bar, grid_display])
            
            # Display
            cv2.imshow(window_name, final_display)
            
            # Handle input
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("🛑 Shutting down...")
                break
            elif key == ord('s'):
                screenshot_name = f"multi_camera_sharp_text_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, final_display)
                print(f"📸 Sharp text screenshot saved: {screenshot_name}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        # Stop all cameras
        print("⏹️  Stopping all cameras...")
        for feed in working_feeds:
            feed.stop()
        
        cv2.destroyAllWindows()
        
        # Statistics
        duration = time.time() - start_time
        print(f"\n📊 Session completed:")
        print(f"   Cameras displayed: {len(working_feeds)}")
        print(f"   Dashboard frames: {frame_count}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Dashboard FPS: {frame_count/duration:.1f}")


if __name__ == '__main__':
    main()
