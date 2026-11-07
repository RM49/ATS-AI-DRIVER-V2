"""
REAL-TIME INFERENCE SCRIPT
==========================
Loads the trained model and predicts steering angles based on 
live game window + telemetry.

USAGE:
1. Start American Truck Simulator.
2. Run this script.
3. Watch the console for predictions vs (Human) Actuals.
"""

import win32gui
import win32ui
import win32con
import ctypes
from PIL import Image
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Concatenate, Dropout, Input
from keras.optimizers import Adam
import numpy as np
import mmap
import struct
import time
from collections import deque
import os

# Filter out TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- CONFIGURATION ---
MODEL_PATH = "model_best.h5" # Path to your saved .h5 file
WINDOW_TITLE = "American Truck Simulator"

# TELEMETRY CONFIG
SHM_NAME = "Local\\SCS_Telemetry_Shared_Mem"
STRUCT_FMT = "Ifffff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

# IMAGE CONFIG (Must match training!)
TARGET_W = 200
TARGET_H = 66
MAX_SPEED = 35.0  # Normalization factor (approx 80 mph)

# CROP CONFIG (Must match training!)
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
            print(f"[Telemetry] Connected!")
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
            return {"speed": speed, "steer": steer}
        except Exception:
            return None

# --- 2. MODEL DEFINITION ---
def create_pilotnet_dual_branch():
    # -- BRANCH A: VISION --
    img_input = Input(shape=(66, 200, 3), name='vision_input')
    
    # NVIDIA Architecture
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(img_input)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    x = Flatten()(x)
    vision_out = Dense(100, activation='relu')(x)
    
    # -- BRANCH B: PHYSICS --
    speed_input = Input(shape=(1,), name='speed_input')
    # Process speed slightly to give it weight
    physics_out = Dense(10, activation='relu')(speed_input)
    
    # -- FUSION --
    # Combine the extracted visual features with the speed info
    merged = Concatenate()([vision_out, physics_out])
    
    # -- CONTROL MODULE --
    x = Dense(50, activation='relu')(merged)
    x = Dropout(0.1)(x) # Helps prevent overfitting
    x = Dense(10, activation='relu')(x)
    
    # Output layer: Tanh ensures range is -1 to 1
    output = Dense(1, activation='linear', name='steering_output')(x)
    
    model = Model(inputs=[img_input, speed_input], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')
    return model

def capture_window(hwnd):
    """Identical to collection script"""
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

def process_image(img):
    """Identical to collection script"""
    w, h = img.size
    left = int(w * CROP_LEFT_PCT)
    right = int(w * CROP_RIGHT_PCT)
    top = int(h * CROP_TOP_PCT)
    bottom = int(h * CROP_BOTTOM_PCT)

    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    
    # Normalize 0-1
    arr = np.array(img_resized.convert('L'), dtype=np.float32)
    return arr / 255.0

def main():
    print("Loading Model...")
    try:
        model = create_pilotnet_dual_branch()
        model.load_weights(MODEL_PATH)
        print("Model Loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    telemetry = TelemetryReader()
    
    print(f"Waiting for {WINDOW_TITLE}...")
    hwnd = 0
    while not hwnd:
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        time.sleep(1)
    print("Game Found. Starting Inference Loop...")

    # --- FRAME BUFFER ---
    # Holds the last 3 processed frames
    # Initialize with zeros so the model doesn't crash on the first frame
    frame_queue = deque(maxlen=3)
    for _ in range(3):
        frame_queue.append(np.zeros((TARGET_H, TARGET_W), dtype=np.float32))

    try:
        while True:
            loop_start = time.time()
            
            # 1. Capture Data
            raw_img = capture_window(hwnd)
            data = telemetry.get_data()
            
            if raw_img and data:
                # 2. Process & Update Queue
                processed_img = process_image(raw_img)
                frame_queue.append(processed_img)
                
                # 3. Stack Frames (Shape: 66, 200, 3)
                # Channels: [Oldest, Middle, Newest]
                input_stack = np.stack(frame_queue, axis=-1)
                
                # 4. Prepare Batch (Model expects batch dimension)
                # Shape becomes (1, 66, 200, 3)
                input_stack = np.expand_dims(input_stack, axis=0)
                
                # 5. Prepare Speed (Shape: 1, 1)
                # NORMALIZE SPEED HERE!
                norm_speed = data['speed'] / MAX_SPEED 
                input_speed = np.array([norm_speed]).reshape(1, 1)
                
                # 6. PREDICT
                # Returns [[steering_value]]
                # Using a dictionary ensures inputs go to the exact right layers
                prediction = model.predict(
                    {
                        'vision_input': input_stack,
                        'speed_input': input_speed
                    }, 
                    verbose=0
                )
                predicted_steer = float(prediction[0][0])
                
                # 7. Print Result
                # We show Predicted vs Actual (Human) to see how well it learned
                error = abs(predicted_steer - data['steer'])
                
                print(f"AI: {predicted_steer:+.2f} | Human: {data['steer']:+.2f} | Speed: {data['speed']:.1f} | Diff: {error:.2f}", end='\r')
            
            # Limit FPS slightly to save CPU (20 FPS)
            # DYNAMIC SLEEP to ensure exactly 10 FPS (0.1s)
            process_duration = time.time() - loop_start
            target_delay = 0.1 # Must match CollectView.py CAPTURE_DELAY
        
            if process_duration < target_delay:
                time.sleep(target_delay - process_duration)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
