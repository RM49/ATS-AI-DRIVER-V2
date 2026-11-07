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
import pygame  # <--- Added Back

# --- CONFIGURATION ---
DATA_DIR = r"..\wsl\data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LOG_FILE = os.path.join(DATA_DIR, "driving_log.csv")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

# TELEMETRY CONFIG (SCS SDK)
SHM_NAME = "Local\\SCS_Telemetry_Shared_Mem"
STRUCT_FMT = "Ifffff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

# VISION CONFIG
TARGET_W = 200
TARGET_H = 66
CAPTURE_DELAY = 0.1

# PREVIEW CONFIG
PREVIEW_SCALE = 3  # Make the window 3x bigger so we can see it

# CROP REGIONS
CROP_TOP_PCT = 0.2   
CROP_BOTTOM_PCT = 0.7 
CROP_LEFT_PCT = 0.2    
CROP_RIGHT_PCT = 1.0

ctypes.windll.user32.SetProcessDPIAware()

class TelemetryReader:
    def __init__(self):
        self.shm = None
        self.connected = False
        self.connect()

    def connect(self):
        try:
            self.shm = mmap.mmap(-1, STRUCT_SIZE, tagname=SHM_NAME)
            self.connected = True
        except FileNotFoundError:
            self.connected = False

    def get_data(self):
        if not self.connected:
            self.connect()
            return None
        try:
            self.shm.seek(0)
            bytes_data = self.shm.read(STRUCT_SIZE)
            timestamp, speed, rpm, steer, throttle, brake = struct.unpack(STRUCT_FMT, bytes_data)
            return {"speed": speed, "steer": steer, "throttle": throttle}
        except Exception:
            self.connected = False
            return None

class DataCollector:
    def __init__(self):
        self.is_recording = False
        self.running = True
        self.telemetry = TelemetryReader()
        self.init_csv()
        
        # --- VISUALIZER SETUP ---
        pygame.init()
        self.display_w = TARGET_W * PREVIEW_SCALE
        self.display_h = TARGET_H * PREVIEW_SCALE
        self.screen = pygame.display.set_mode((self.display_w, self.display_h))
        pygame.display.set_caption("AI Vision Feed (200x66)")

    def init_csv(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'image_filename', 'steering', 'throttle', 'speed'])

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
        """Returns Grayscale PIL Image for saving"""
        w, h = img.size
        left = int(w * CROP_LEFT_PCT)
        right = int(w * CROP_RIGHT_PCT)
        top = int(h * CROP_TOP_PCT)
        bottom = int(h * CROP_BOTTOM_PCT)

        img_cropped = img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        return img_resized.convert('L')

    def update_preview(self, pil_img):
        """Updates the Pygame window"""
        # 1. Convert Grayscale ('L') to RGB for Pygame
        # (Pygame doesn't like 8-bit grayscale strings directly)
        rgb_img = pil_img.convert("RGB")
        
        # 2. Create Pygame Surface
        mode = params = rgb_img.mode
        size = rgb_img.size
        data = rgb_img.tobytes()
        py_image = pygame.image.fromstring(data, size, mode)
        
        # 3. Scale up for visibility
        py_image = pygame.transform.scale(py_image, (self.display_w, self.display_h))
        
        # 4. Draw
        self.screen.blit(py_image, (0, 0))
        
        # 5. Visual Indicator for Recording
        if self.is_recording:
            pygame.draw.circle(self.screen, (255, 0, 0), (20, 20), 10) # Red Dot
        
        pygame.display.flip()

    def save_sample(self, img, data):
        timestamp = time.time()
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        img.save(filepath)

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, filename, f"{data['steer']:.4f}", f"{data['throttle']:.4f}", f"{data['speed']:.4f}"])

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
        window_title = "American Truck Simulator"
        hwnd = 0
        while not hwnd:
            hwnd = win32gui.FindWindow(None, window_title)
            if not hwnd:
                print("Waiting for game window...", end='\r')
                time.sleep(1)
        
        print(f"\nFound {window_title}!")
        print("Waiting for Telemetry...")
        while not self.telemetry.connected:
            self.telemetry.connect()
            time.sleep(1)

        print("READY. Press 'c' to record, 'q' to quit.")
        listener = self.keyboard_listener()
        last_capture_time = 0

        while self.running:
            # Handle Pygame Events (prevents "Not Responding" errors)
            pygame.event.pump()
            
            loop_start = time.time()
            data = self.telemetry.get_data()
            
            if not data:
                print("Telemetry lost...", end='\r')
                time.sleep(1)
                continue

            # Capture & Process
            if (loop_start - last_capture_time) > CAPTURE_DELAY:
                raw_img = self.capture_window(hwnd)
                
                if raw_img:
                    final_img = self.process_image(raw_img)
                    
                    # Update the live view
                    self.update_preview(final_img)
                    
                    if self.is_recording:
                        self.save_sample(final_img, data)
                        print(f"REC | Steer: {data['steer']:.2f}", end='\r')
                    else:
                        print(f"IDLE| Steer: {data['steer']:.2f}", end='\r')
                    
                    last_capture_time = time.time()
            
            # Small sleep to prevent maxing out CPU
            time.sleep(0.01)

        listener.stop()
        pygame.quit() # <--- Cleanup
        print("\nSession saved.")

if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
