#!/usr/bin/env python3
"""
Debug script for tower health detection issues.

This script helps debug common problems with the tower health detector:
1. Region extraction issues
2. OCR problems
3. Health bar detection failures
4. Configuration problems
5. Screen resolution mismatches
"""

import cv2
import numpy as np
import sys
import os
import argparse
from typing import Optional, Tuple

# Add the pyclashbot module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyclashbot'))

from pyclashbot.detection.tower_health_detection import TowerHealthDetector, TowerHealth
from pyclashbot.config.movement_bot_config import get_movement_bot_config


class TowerHealthDebugger:
    """Debug tool for tower health detection."""
    
    def __init__(self):
        self.detector = TowerHealthDetector()
        self.config = get_movement_bot_config()
        
    def debug_region_extraction(self, frame: np.ndarray, save_debug_images: bool = True) -> None:
        """Debug tower region extraction."""
        print("=== Debugging Region Extraction ===")
        
        height, width = frame.shape[:2]
        print(f"Frame dimensions: {width}x{height}")
        
        for region_name, (x_inches, y_inches, w_inches, h_inches) in self.detector.tower_regions.items():
            print(f"\nRegion: {region_name}")
            print(f"  Inches: x={x_inches}, y={y_inches}, w={w_inches}, h={h_inches}")
            
            # Convert to pixels
            screen_width_inches = 3.5
            screen_height_inches = 6.5
            
            x1 = int((x_inches / screen_width_inches) * width)
            y1 = int((y_inches / screen_height_inches) * height)
            x2 = int(((x_inches + w_inches) / screen_width_inches) * width)
            y2 = int(((y_inches + h_inches) / screen_height_inches) * height)
            
            print(f"  Pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Extract region
            region = self.detector.extract_tower_region(frame, region_name)
            
            if region.size == 0:
                print(f"  ERROR: Empty region extracted!")
                continue
                
            print(f"  Region size: {region.shape}")
            
            if save_debug_images:
                debug_path = f"debug_region_{region_name}.png"
                cv2.imwrite(debug_path, region)
                print(f"  Saved debug image: {debug_path}")
    
    def debug_ocr_processing(self, frame: np.ndarray, save_debug_images: bool = True) -> None:
        """Debug OCR processing for each region."""
        print("\n=== Debugging OCR Processing ===")
        
        for region_name in self.detector.tower_regions:
            print(f"\nRegion: {region_name}")
            
            region = self.detector.extract_tower_region(frame, region_name)
            if region.size == 0:
                print(f"  ERROR: Empty region, skipping OCR")
                continue
            
            # Test OCR processing
            try:
                health_value = self.detector.extract_health_text(region)
                print(f"  OCR result: {health_value}")
                
                if save_debug_images:
                    # Save original region
                    cv2.imwrite(f"debug_ocr_original_{region_name}.png", region)
                    
                    # Save processed region for OCR
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    if np.mean(binary) < 127:
                        binary = cv2.bitwise_not(binary)
                    
                    kernel = np.ones((2, 2), np.uint8)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    
                    cv2.imwrite(f"debug_ocr_processed_{region_name}.png", binary)
                    print(f"  Saved OCR debug images for {region_name}")
                    
            except Exception as e:
                print(f"  ERROR in OCR processing: {e}")
    
    def debug_health_bar_detection(self, frame: np.ndarray, save_debug_images: bool = True) -> None:
        """Debug health bar detection."""
        print("\n=== Debugging Health Bar Detection ===")
        
        for region_name in self.detector.tower_regions:
            print(f"\nRegion: {region_name}")
            
            region = self.detector.extract_tower_region(frame, region_name)
            if region.size == 0:
                print(f"  ERROR: Empty region, skipping health bar detection")
                continue
            
            # Test health bar detection
            try:
                health_percentage = self.detector.detect_health_bar(region)
                print(f"  Health bar percentage: {health_percentage}")
                
                # Test health bar presence detection
                has_health_bar = self.detector._detect_health_bar_presence(region)
                print(f"  Health bar present: {has_health_bar}")
                
                if save_debug_images:
                    # Save HSV analysis
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    cv2.imwrite(f"debug_hsv_{region_name}.png", hsv)
                    
                    # Create color masks
                    for color_name, (lower, upper) in self.detector.health_bar_color_ranges.items():
                        lower = np.array(lower)
                        upper = np.array(upper)
                        mask = cv2.inRange(hsv, lower, upper)
                        cv2.imwrite(f"debug_mask_{color_name}_{region_name}.png", mask)
                    
                    print(f"  Saved health bar debug images for {region_name}")
                    
            except Exception as e:
                print(f"  ERROR in health bar detection: {e}")
    
    def debug_full_detection(self, frame: np.ndarray, save_debug_images: bool = True) -> None:
        """Debug full tower health detection process."""
        print("\n=== Debugging Full Detection Process ===")
        
        try:
            tower_health = self.detector.detect_all_tower_health(frame)
            
            print(f"Detection results:")
            print(f"  Own Left Tower: {tower_health.left_tower_health}")
            print(f"  Own Right Tower: {tower_health.right_tower_health}")
            print(f"  Enemy Left Tower: {tower_health.enemy_left_tower_health}")
            print(f"  Enemy Right Tower: {tower_health.enemy_right_tower_health}")
            print(f"  Overall Confidence: {tower_health.confidence}")
            print(f"  Timestamp: {tower_health.timestamp}")
            
            # Check tower damage status
            print(f"\nTower damage status:")
            for region_name, damaged in self.detector.towers_damaged.items():
                print(f"  {region_name}: {'Damaged' if damaged else 'Not Damaged'}")
            
            # Check initial health values
            print(f"\nInitial health values:")
            for region_name, health in self.detector.initial_health.items():
                print(f"  {region_name}: {health}")
            
            # Check last health values
            print(f"\nLast health values:")
            for region_name, health in self.detector.last_health.items():
                print(f"  {region_name}: {health}")
            
            if save_debug_images:
                # Create visualization
                vis_frame = self.detector.create_health_visualization(frame, tower_health)
                cv2.imwrite("debug_full_detection_visualization.png", vis_frame)
                print(f"  Saved full detection visualization")
                
        except Exception as e:
            print(f"  ERROR in full detection: {e}")
            import traceback
            traceback.print_exc()
    
    def debug_configuration(self) -> None:
        """Debug configuration settings."""
        print("\n=== Debugging Configuration ===")
        
        print(f"Tower health enabled: {self.config.tower_health_enabled}")
        print(f"OCR enabled: {self.config.enable_ocr}")
        print(f"Health bar detection enabled: {self.config.enable_health_bar_detection}")
        print(f"OCR confidence threshold: {self.config.ocr_confidence_threshold}")
        print(f"Health bar confidence threshold: {self.config.health_bar_confidence_threshold}")
        print(f"Tower health visualization enabled: {self.config.enable_tower_health_visualization}")
        print(f"Show tower health: {self.config.show_tower_health}")
        
        # Check OCR configuration
        print(f"\nOCR configuration: {self.detector.ocr_config}")
        
        # Check health bar color ranges
        print(f"\nHealth bar color ranges:")
        for color, (lower, upper) in self.detector.health_bar_color_ranges.items():
            print(f"  {color}: {lower} - {upper}")
    
    def debug_dependencies(self) -> None:
        """Debug external dependencies."""
        print("\n=== Debugging Dependencies ===")
        
        # Check OpenCV
        try:
            print(f"OpenCV version: {cv2.__version__}")
        except Exception as e:
            print(f"ERROR: OpenCV not available: {e}")
        
        # Check Tesseract
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
        except Exception as e:
            print(f"ERROR: Tesseract not available: {e}")
        
        # Check NumPy
        try:
            print(f"NumPy version: {np.__version__}")
        except Exception as e:
            print(f"ERROR: NumPy not available: {e}")
    
    def run_full_debug(self, frame: np.ndarray, save_debug_images: bool = True) -> None:
        """Run complete debugging suite."""
        print("Starting Tower Health Detection Debug Session")
        print("=" * 50)
        
        self.debug_dependencies()
        self.debug_configuration()
        self.debug_region_extraction(frame, save_debug_images)
        self.debug_ocr_processing(frame, save_debug_images)
        self.debug_health_bar_detection(frame, save_debug_images)
        self.debug_full_detection(frame, save_debug_images)
        
        print("\n" + "=" * 50)
        print("Debug session completed!")
        
        if save_debug_images:
            print("Debug images saved to current directory")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Debug tower health detection")
    parser.add_argument("--image", "-i", help="Path to test image file")
    parser.add_argument("--no-save", action="store_true", help="Don't save debug images")
    parser.add_argument("--region", "-r", help="Debug specific region only")
    
    args = parser.parse_args()
    
    debugger = TowerHealthDebugger()
    
    if args.image:
        # Load test image
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Could not load image {args.image}")
            return
        
        print(f"Loaded test image: {args.image}")
        print(f"Image dimensions: {frame.shape}")
        
        save_images = not args.no_save
        
        if args.region:
            # Debug specific region only
            if args.region in debugger.detector.tower_regions:
                print(f"Debugging region: {args.region}")
                region = debugger.detector.extract_tower_region(frame, args.region)
                if region.size > 0:
                    cv2.imwrite(f"debug_region_{args.region}.png", region)
                    print(f"Saved region image: debug_region_{args.region}.png")
                else:
                    print(f"ERROR: Empty region for {args.region}")
            else:
                print(f"ERROR: Unknown region {args.region}")
                print(f"Available regions: {list(debugger.detector.tower_regions.keys())}")
        else:
            # Run full debug
            debugger.run_full_debug(frame, save_images)
    else:
        # Just show configuration and dependencies
        debugger.debug_dependencies()
        debugger.debug_configuration()
        print("\nTo debug with an image, use: python debug_tower_health.py --image <path_to_image>")


if __name__ == "__main__":
    main()
