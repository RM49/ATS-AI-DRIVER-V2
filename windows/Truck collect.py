"""
DATA COLLECTION SCRIPT (TELEMETRY + VISION)
===========================================

INTEGRATION:
- Visual: Captures game window (PilotNet Standard: 200x66)
- Physics: Reads shared memory from SCS SDK (Speed, Steering, Throttle)

TARGET MODEL:
- Late Fusion Architecture (CNN + Dense)
- Inputs: Image Stack (Vision) + Speed (Physics)
- Output: Steering Value
"""

import win32gui
import win32ui
import win32con
import ctypes
from PIL import Image
import time
import numpy as np
import os
import csv
import mmap
import struct
from pynput import keyboard

# --- CONFIGURATION ---
DATA_DIR = r"..\wsl\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LOG_FILE = os.path.join(DATA_DIR, "driving_log.csv")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# TELEMETRY CONFIG (SCS SDK)
SHM_NAME = "Local\\SCS_Telemetry_Shared_Mem"
# Struct: unsigned int, float, float, float, float, float
# (Timestamp, Speed, RPM, Steer, Throttle, Brake)
STRUCT_FMT = "Ifffff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

# VISION CONFIG (NVIDIA Standard)
TARGET_W = 200
TARGET_H = 66
CAPTURE_DELAY = 0.1  # 10 Hz recording

# CROP REGIONS (Percentages)
CROP_TOP_PCT = 0.35    
CROP_BOTTOM_PCT = 0.85 
CROP_LEFT_PCT = 0.0    
CROP_RIGHT_PCT = 1.0

# DPI Awareness
ctypes.windll.user32.SetProcessDPIAware()

class TelemetryReader:
    """Handles connection to SCS Shared Memory"""
    def __init__(self):
        self.shm = None
        self.connected = False
        self.connect()

    def connect(self):
        try:
            # -1 = System Paging File (RAM)
            self.shm = mmap.mmap(-1, STRUCT_SIZE, tagname=SHM_NAME)
            self.connected = True
            print(f"[Telemetry] Connected to {SHM_NAME}")
        except FileNotFoundError:
            self.connected = False
            # We don't spam print here to avoid console flooding

    def get_data(self):
        """Returns dict: {speed, steer, throttle} or None"""
        if not self.connected:
            self.connect()
            return None

        try:
            self.shm.seek(0)
            bytes_data = self.shm.read(STRUCT_SIZE)
            timestamp, speed, rpm, steer, throttle, brake = struct.unpack(STRUCT_FMT, bytes_data)
            
            return {
                "speed": speed,      # m/s
                "steer": steer,      # -1.0 to 1.0
                "throttle": throttle # 0.0 to 1.0
            }
        except Exception:
            self.connected = False # Reset connection on error
            return None

class DataCollector:
    def __init__(self):
        self.is_recording = False
        self.running = True
        self.telemetry = TelemetryReader()
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'center', 'steering', 'throttle', 'speed'])

    def capture_window(self, hwnd):
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            w = right - left
            h = bottom - top

            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(save_bitmap)

            # PrintWindow with PW_RENDERFULLCONTENT (2)
            ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)

            bmp_info = save_bitmap.GetInfo()
            bmp_str = save_bitmap.GetBitmapBits(True)

            img = Image.frombuffer(
                'RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']),
                bmp_str, 'raw', 'BGRX', 0, 1)

            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            return img
        except Exception:
            return None

    def process_image(self, img):
        w, h = img.size
        left = int(w * CROP_LEFT_PCT)
        right = int(w * CROP_RIGHT_PCT)
        top = int(h * CROP_TOP_PCT)
        bottom = int(h * CROP_BOTTOM_PCT)

        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        return img_resized.convert('L') # Grayscale

    def save_sample(self, img, data):
        timestamp = time.time()
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)

        img.save(filepath)

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            # CSV Format: [Time, Path, Steer, Throttle, Speed]
            writer.writerow([
                timestamp, 
                filename, 
                f"{data['steer']:.4f}", 
                f"{data['throttle']:.4f}", 
                f"{data['speed']:.4f}"
            ])

    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'c':
                    self.is_recording = not self.is_recording
                    print(f"\nRecording: {'ON' if self.is_recording else 'OFF'}")
                if key.char == 'q':
                    self.running = False
            except AttributeError:
                pass
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener

    def run(self):
        print("Searching for 'American Truck Simulator'...")
        # Make sure this matches your game window title exactly
        window_title = "American Truck Simulator" 
        
        # Wait for window
        hwnd = 0
        while not hwnd:
            hwnd = win32gui.FindWindow(None, window_title)
            if not hwnd:
                print("Waiting for game window...", end='\r')
                time.sleep(1)
        
        print(f"\nFound {window_title}!")
        print("Waiting for Telemetry connection...")
        
        # Wait for valid telemetry before starting loop
        while not self.telemetry.connected:
            self.telemetry.connect()
            time.sleep(1)

        print("READY. Press 'c' to record, 'q' to quit.")
        listener = self.keyboard_listener()
        
        last_capture_time = 0

        while self.running:
            loop_start = time.time()
            
            # 1. Get Physics Data (Real time)
            data = self.telemetry.get_data()
            
            if not data:
                # If telemetry fails (game paused/closed), we wait
                print("Telemetry lost. Waiting...", end='\r')
                time.sleep(1)
                continue

            # 2. Recording Logic
            if self.is_recording and (loop_start - last_capture_time) > CAPTURE_DELAY:
                
                raw_img = self.capture_window(hwnd)
                
                if raw_img:
                    final_img = self.process_image(raw_img)
                    self.save_sample(final_img, data)
                    last_capture_time = time.time()
                    
                    status = f"REC | Steer: {data['steer']:.2f} | Speed: {data['speed']:.1f} m/s"
                    print(status, end='\r')
            
            elif not self.is_recording:
                # Just show telemetry to prove it's working
                status = f"IDLE| Steer: {data['steer']:.2f} | Speed: {data['speed']:.1f} m/s"
                print(status, end='\r')
                time.sleep(0.05)

        listener.stop()
        print("\nSession saved.")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
