import win32gui
import win32ui
import win32con
import ctypes
from PIL import Image
import pygame  # Added for visualization
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Concatenate, Dropout, Input
from keras.optimizers import Adam
import numpy as np
import mmap
import struct
import time
from collections import deque
import os
import math

# Filter out TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# --- CONFIGURATION ---
MODEL_PATH = "model_best.h5" 
WINDOW_TITLE = "American Truck Simulator"
SHM_NAME = "Local\\SCS_Telemetry_Shared_Mem"
STRUCT_FMT = "Ifffff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

TARGET_W = 200
TARGET_H = 66
MAX_SPEED = 35.0
PREVIEW_SCALE = 3 # Match your collector scale

# CROP CONFIG
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

def create_pilotnet_dual_branch():
    img_input = Input(shape=(66, 200, 3), name='vision_input')
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(img_input)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    vision_out = Dense(100, activation='relu')(x)
    speed_input = Input(shape=(1,), name='speed_input')
    physics_out = Dense(10, activation='relu')(speed_input)
    merged = Concatenate()([vision_out, physics_out])
    x = Dense(50, activation='relu')(merged)
    x = Dropout(0.1)(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(1, activation='linear', name='steering_output')(x)
    model = Model(inputs=[img_input, speed_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')
    return model

def capture_window(hwnd):
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w, h = right - left, bottom - top
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(save_bitmap)
        ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
        bmp_info = save_bitmap.GetInfo()
        bmp_str = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), bmp_str, 'raw', 'BGRX', 0, 1)
        win32gui.DeleteObject(save_bitmap.GetHandle()); save_dc.DeleteDC(); mfc_dc.DeleteDC(); win32gui.ReleaseDC(hwnd, hwnd_dc)
        return img
    except Exception: return None

def process_image(img):
    w, h = img.size
    img_cropped = img.crop((int(w*CROP_LEFT_PCT), int(h*CROP_TOP_PCT), int(w*CROP_RIGHT_PCT), int(h*CROP_BOTTOM_PCT)))
    img_resized = img_cropped.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
    arr = np.array(img_resized.convert('L'), dtype=np.float32)
    return arr / 255.0, img_resized # Return both normalized array and the PIL image for display

def draw_steering_line(screen, steer_val):
    """Draws a green line indicating predicted steering"""
    # Origin scaled to match preview
    # User specified: 75px from left, 0 from bottom (in 200x66 space)
    origin_x = 75 * PREVIEW_SCALE
    origin_y = (TARGET_H) * PREVIEW_SCALE
    
    line_length = 40 * PREVIEW_SCALE
    # steer_val 1 is LEFT (negative angle in screen space), -1 is RIGHT (positive angle)
    # We map 1.0 to roughly -45 degrees and -1.0 to 45 degrees
    angle_rad = steer_val * (math.pi / 4) 
    
    # Calculate end point: X is shifted by steering, Y goes UP (negative)
    # Note: steering 1 (left) should decrease X. So we subtract.
    end_x = origin_x - (math.sin(angle_rad) * line_length)
    end_y = origin_y - (math.cos(angle_rad) * line_length)
    
    pygame.draw.line(screen, (0, 255, 0), (origin_x, origin_y), (end_x, end_y), 5)

def main():
    # --- PYGAME SETUP ---
    pygame.init()
    display_w, display_h = TARGET_W * PREVIEW_SCALE, TARGET_H * PREVIEW_SCALE
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("AI Prediction Overlay")

    print("Loading Model...")
    model = create_pilotnet_dual_branch()
    model.load_weights(MODEL_PATH)
    telemetry = TelemetryReader()
    
    hwnd = 0
    while not hwnd:
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        time.sleep(1)

    frame_queue = deque(maxlen=3)
    for _ in range(3): frame_queue.append(np.zeros((TARGET_H, TARGET_W), dtype=np.float32))

    try:
        while True:
            pygame.event.pump()
            loop_start = time.time()
            raw_img = capture_window(hwnd)
            data = telemetry.get_data()
            
            if raw_img and data:
                # 1. Process
                processed_arr, display_img = process_image(raw_img)
                frame_queue.append(processed_arr)
                
                # 2. Predict
                input_stack = np.expand_dims(np.stack(frame_queue, axis=-1), axis=0)
                input_speed = np.array([data['speed'] / MAX_SPEED]).reshape(1, 1)
                
                prediction = model.predict({'vision_input': input_stack, 'speed_input': input_speed}, verbose=0)
                predicted_steer = float(prediction[0][0])
                
                # 3. Visualize
                # Convert PIL to Pygame Surface
                rgb_img = display_img.convert("RGB")
                py_image = pygame.image.fromstring(rgb_img.tobytes(), rgb_img.size, rgb_img.mode)
                py_image = pygame.transform.scale(py_image, (display_w, display_h))
                
                screen.blit(py_image, (0, 0))
                draw_steering_line(screen, predicted_steer)
                pygame.display.flip()
                
                print(f"AI: {predicted_steer:+.2f} | Human: {data['steer']:+.2f} | Diff: {abs(predicted_steer - data['steer']):.2f}", end='\r')
            
            process_duration = time.time() - loop_start
            target_delay = 0.1 
            if process_duration < target_delay:
                time.sleep(target_delay - process_duration)
            
    except KeyboardInterrupt:
        print("\nStopped.")
        pygame.quit()

if __name__ == "__main__":
    main()