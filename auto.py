#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Camera Discovery and Launch Script

Gets camera list from API endpoint and launches the blur detection dashboard.
"""

import requests
import subprocess
import sys
import json


def get_cameras_from_api(base_url):
    """
    Get camera list from the API endpoint
    
    Args:
        base_url: Base URL of the camera interface
        
    Returns:
        list: Camera IP addresses from API
    """
    camera_ips = []
    try:
        print("🚀 Getting camera list from API...")
        
        # API endpoint for camera list
        api_url = f"{base_url.rstrip('/')}/api.php?v=2&action=check_box_state"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*'
        }
        
        print(f"📡 Querying: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✅ API responded successfully")
            
            try:
                data = response.json()
                print(f"📄 API Response: {json.dumps(data, indent=2)}")
                
                # Extract cameras from the 'boxes' array
                if 'boxes' in data and isinstance(data['boxes'], list):
                    print(f"📍 Found {len(data['boxes'])} camera boxes in API response")
                    
                    for box in data['boxes']:
                        # Get IP from 'ip' field
                        ip = box.get('ip')
                        if ip:
                            camera_ips.append(ip)
                            # Show camera info
                            dispname = box.get('dispname', 'Unknown')
                            master = box.get('master', 0)
                            order = box.get('order', 0)
                            master_text = " (MASTER)" if master else ""
                            print(f"   📷 {ip} - {dispname}{master_text} (order: {order})")
                        else:
                            print(f"   ⚠️ Box missing IP: {box}")
                else:
                    print("❌ No 'boxes' array found in API response")
                    return []
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse API response as JSON: {e}")
                print(f"Raw response: {response.text[:200]}...")
                return []
        else:
            print(f"❌ API endpoint returned status code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return []
                
    except Exception as e:
        print(f"❌ API request failed: {e}")
        return []
    
    return camera_ips


def test_camera_connection(camera_ip, port="9080"):
    """
    Test if a camera is reachable
    
    Args:
        camera_ip: Camera IP address
        port: Camera port (default 9080)
        
    Returns:
        bool: True if camera is reachable
    """
    try:
        test_urls = [
            f"http://{camera_ip}:{port}/?action=snapshot",
            f"http://{camera_ip}:{port}/?action=stream",
            f"http://{camera_ip}:{port}/",
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    return True
            except:
                continue
                
        return False
        
    except Exception:
        return False


def launch_camera_dashboard(camera_ips, port="9080"):
    """
    Launch the camera dashboard with discovered cameras
    
    Args:
        camera_ips: List of camera IP addresses
        port: Port to use for cameras (default 9080)
    """
    if not camera_ips:
        print("❌ No cameras found. Exiting.")
        return
    
    # Build camera endpoints with port
    camera_endpoints = [f"{ip}:{port}" for ip in camera_ips]
    
    # Filter to only reachable cameras
    reachable_cameras = []
    print("\n🔍 Testing camera connections...")
    
    for i, ip in enumerate(camera_ips):
        endpoint = camera_endpoints[i]
        print(f"   Testing {endpoint}...", end=' ')
        reachable_cameras.append(endpoint)
    
    if not reachable_cameras:
        print("❌ No reachable cameras found. Exiting.")
        print("💡 Check if cameras are powered on and network is accessible")
        return
    
    # Build command line arguments
    cmd = ['camera.exe', '-c'] + reachable_cameras
    
    print(f"\n🚀 Launching camera dashboard with {len(reachable_cameras)} cameras:")
    for i, camera in enumerate(reachable_cameras, 1):
        print(f"   📷 Camera {i}: {camera}")
    
    print(f"\n📋 Command: {' '.join(cmd)}")
    print("\n" + "="*50)
    
    try:
        # Launch the camera dashboard
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch camera dashboard: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except FileNotFoundError:
        print("❌ camera.py not found in current directory")
        print("💡 Make sure you're running this script from the blur detection directory")


def main():
    """Main function"""
    print("🤖 Auto Camera Discovery and Launch")
    print("=" * 50) 
    print("🚀 METHOD: API endpoint (api.php?v=2&action=check_box_state)")
    print("🎯 SIMPLE: Get JSON → Extract IPs → Launch camera.py")
    print("")
    
    # Default base URL
    base_url = "http://169.254.200.200/"
    port = "9080"  # Default camera port
    
    # Allow custom URL via command line
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
        if not base_url.startswith('http'):
            base_url = f"http://{base_url}"
        if not base_url.endswith('/'):
            base_url += '/'
    
    # Allow custom port via command line  
    if len(sys.argv) > 2:
        port = sys.argv[2]
    
    print(f"🎯 Target URL: {base_url}")
    print(f"🔌 Camera Port: {port}")
    
    # Get camera IPs from API
    camera_ips = get_cameras_from_api(base_url)
    
    if not camera_ips:
        print("❌ No cameras discovered")
        print("💡 Troubleshooting tips:")
        print("   - Check if the URL is accessible in a web browser")
        print("   - Verify the API endpoint responds:")
        print(f"     {base_url}api.php?v=2&action=check_box_state")
        print("   - Try with a different URL: python auto.py http://your-ip/")
        return
    
    # Launch dashboard
    launch_camera_dashboard(camera_ips, port)


if __name__ == '__main__':
    main()
