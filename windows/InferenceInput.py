import win32gui
import win32ui
import win32con
import ctypes
from PIL import Image
import pygame
import vgamepad as vg
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
import csv
from pynput import keyboard

# --- CONFIGURATION ---
# 1. PATHS (Matches CollectView.py)
DATA_DIR = r"..\wsl\data"
# DATA_DIR = "recoverData/"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LOG_FILE = os.path.join(DATA_DIR, "driving_log.csv")

MODEL_PATH = "model_best.h5"
WINDOW_TITLE = "American Truck Simulator"

# 2. TELEMETRY
SHM_NAME = "Local\\SCS_Telemetry_Shared_Mem"
STRUCT_FMT = "Ifffff"
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

# 3. VISION
TARGET_W = 200
TARGET_H = 66
MAX_SPEED = 35.0
PREVIEW_SCALE = 3
CROP_TOP_PCT = 0.2
CROP_BOTTOM_PCT = 0.7
CROP_LEFT_PCT = 0.2
CROP_RIGHT_PCT = 1.0

# 4. CONTROL (GREEDY PID)
CONTROL_FREQ = 60
INFERENCE_INTERVAL = 0.1
PID_KP = 12.0
PID_KI = 4.0
PID_KD = 0.5
MIN_ACTUATION = 0.1

os.makedirs(IMAGES_DIR, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ctypes.windll.user32.SetProcessDPIAware()

# --- HELPER CLASSES ---

class DataCollector:
    """Handles saving data to the main training set"""
    def __init__(self):
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'image_filename', 'steering', 'throttle', 'speed'])

    def save_sample(self, img, data):
        """Saves current state to the main dataset"""
        timestamp = time.time()
        # Prefix 'int_' so you know it came from intervention
        filename = f"int_{timestamp}.jpg" 
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Save Image
        img.save(filepath)

        # Save CSV Entry
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, filename, f"{data['steer']:.4f}", f"{data['throttle']:.4f}", f"{data['speed']:.4f}"])

class KeyMonitor:
    """Tracks if A or D are currently held down"""
    def __init__(self):
        self.steering_active = False
        self.keys_down = set()
        
        def on_press(key):
            try:
                if key.char in ['a', 'd']:
                    self.keys_down.add(key.char)
                    self.steering_active = True
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in ['a', 'd']:
                    self.keys_down.discard(key.char)
                    if not self.keys_down:
                        self.steering_active = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

class TelemetryReader:
    def __init__(self):
        self.shm = None
        self.connect()

    def connect(self):
        try:
            self.shm = mmap.mmap(-1, STRUCT_SIZE, tagname=SHM_NAME)
        except FileNotFoundError: pass

    def get_data(self):
        if not self.shm: self.connect(); return None
        try:
            self.shm.seek(0)
            data = struct.unpack(STRUCT_FMT, self.shm.read(STRUCT_SIZE))
            # timestamp, speed, rpm, steer, throttle, brake
            return {"speed": data[1], "steer": data[3], "throttle": data[4]}
        except: return None

# --- MAIN DRIVER ---

class InferenceDriver:
    def __init__(self):
        self.ai_active = False
        self.running = True
        
        print("Initializing Virtual Gamepad...")
        try: self.gamepad = vg.VX360Gamepad()
        except Exception as e: print(f"Gamepad Error: {e}"); exit()
            
        print("Loading Model...")
        self.model = self.load_model()
        self.model.load_weights(MODEL_PATH)
        
        self.telemetry = TelemetryReader()
        self.collector = DataCollector()
        self.key_mon = KeyMonitor()

        # Visualization
        pygame.init()
        self.dw, self.dh = TARGET_W * PREVIEW_SCALE, TARGET_H * PREVIEW_SCALE
        self.screen = pygame.display.set_mode((self.dw, self.dh))
        pygame.display.set_caption("AI Driver + Intervention Train")
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)

    def load_model(self):
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

    def capture_window(self, hwnd):
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
            img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), save_bitmap.GetBitmapBits(True), 'raw', 'BGRX', 0, 1)
            win32gui.DeleteObject(save_bitmap.GetHandle()); save_dc.DeleteDC(); mfc_dc.DeleteDC(); win32gui.ReleaseDC(hwnd, hwnd_dc)
            return img
        except: return None

    def process(self, img):
        w, h = img.size
        img_c = img.crop((int(w*CROP_LEFT_PCT), int(h*CROP_TOP_PCT), int(w*CROP_RIGHT_PCT), int(h*CROP_BOTTOM_PCT)))
        img_r = img_c.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        # Return normalized array AND the resized PIL image (grayscale)
        img_gray = img_r.convert('L')
        return np.array(img_gray, dtype=np.float32) / 255.0, img_gray

    def update_display(self, pil_img_gray, ai_steer, real_steer, input_val, status):
        # 1. Background Image
        rgb_img = pil_img_gray.convert("RGB")
        py_image = pygame.image.fromstring(rgb_img.tobytes(), rgb_img.size, rgb_img.mode)
        py_image = pygame.transform.scale(py_image, (self.dw, self.dh))
        self.screen.blit(py_image, (0, 0))

        # 2. Steering Lines
        center_x, bottom_y = self.dw // 2, self.dh
        line_len = self.dh - 10
        
        def draw_line(val, color, width):
            angle = val * (math.pi / 4)
            end_x = center_x - (math.sin(angle) * line_len)
            end_y = bottom_y - (math.cos(angle) * line_len)
            pygame.draw.line(self.screen, color, (center_x, bottom_y), (end_x, end_y), width)

        draw_line(ai_steer, (0, 255, 0), 4)   # GREEN: AI Target
        draw_line(real_steer, (0, 0, 255), 2) # BLUE: Real Steering

        # 3. Status Indicator (The Dot)
        if status == "REC":
            color = (255, 0, 0) # RED = RECORDING/INTERVENTION
            text = f"REC (Human)"
        elif status == "AI":
            color = (0, 255, 0) # GREEN = AI DRIVING
            text = "AI AUTO"
        else:
            color = (100, 100, 100) # GRAY = OFF
            text = "MANUAL"

        pygame.draw.circle(self.screen, color, (25, 25), 12)
        
        # 4. Text Overlay
        text_surf = self.font.render(f"{text} | In: {input_val:.2f}", True, (255, 255, 255))
        # Add a black drop shadow for readability
        shadow_surf = self.font.render(f"{text} | In: {input_val:.2f}", True, (0, 0, 0))
        self.screen.blit(shadow_surf, (46, 16)) 
        self.screen.blit(text_surf, (45, 15))

        pygame.display.flip()

    def run(self):
        hwnd = 0
        while not hwnd:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            time.sleep(1)
            print("Waiting for game window...", end='\r')
        print("\nWindow Found!")

        # Global Key Listener for Toggle
        def on_press(key):
            try:
                if key.char == 'c': self.ai_active = not self.ai_active
                if key.char == 'q': self.running = False
            except: pass
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        # Buffers & Clocks
        frame_queue = deque(maxlen=3)
        for _ in range(3): frame_queue.append(np.zeros((TARGET_H, TARGET_W), dtype=np.float32))
        clock = pygame.time.Clock()
        
        last_inf = 0
        target_steer = 0.0
        current_input = 0.0
        display_img_gray = None # Holds the PIL image for display/saving

        print("READY. Press 'c' to Toggle AI. Press 'a'/'d' to Intervene.")

        while self.running:
            pygame.event.pump()
            dt = clock.tick(CONTROL_FREQ) / 1000.0
            cur_time = time.time()
            
            data = self.telemetry.get_data()
            if not data: time.sleep(0.1); continue

            # --- 1. VISION & INFERENCE ---
            if (cur_time - last_inf) > INFERENCE_INTERVAL:
                raw_img = self.capture_window(hwnd)
                if raw_img:
                    proc_arr, display_img_gray = self.process(raw_img)
                    frame_queue.append(proc_arr)
                    
                    inp_stack = np.expand_dims(np.stack(frame_queue, axis=-1), axis=0)
                    inp_speed = np.array([data['speed'] / MAX_SPEED]).reshape(1, 1)
                    
                    pred = self.model.predict({'vision_input': inp_stack, 'speed_input': inp_speed}, verbose=0)
                    target_steer = float(pred[0][0])
                    last_inf = cur_time

            # --- 2. CONTROL & INTERVENTION LOGIC ---
            status = "OFF"
            
            if self.ai_active:
                if self.key_mon.steering_active:
                    # >>> INTERVENTION MODE <<<
                    status = "REC"
                    
                    # A. Kill Virtual Input (Let Human Steer)
                    self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
                    self.gamepad.update()
                    current_input = 0.0
                    
                    # B. Save Data (Only if we have a fresh image)
                    if display_img_gray:
                        self.collector.save_sample(display_img_gray, data)
                
                else:
                    # >>> AI MODE <<<
                    status = "AI"
                    
                    # Calculate the difference
                    error = target_steer - data['steer']
                    
                    # CONFIG: Tune these to change how it drives
                    TOLERANCE = 0.004       # If error is within ?%, drive straight (stops shaking)
                    AGGRESSIVE_LIMIT = 0.05 # If error is bigger than 15%, use HARD turn
                    
                    HARD_TURN_VAL = 0.75    # Maximum force for big turns
                    SOFT_TURN_VAL = 0.38   # Gentle force for small adjustments
                    
                    # LOGIC
                    if abs(error) < TOLERANCE:
                        # We are close enough, center the wheel
                        x_val = 0.0
                        
                    elif error < 0:
                        # Target is "Left" of Actual (based on your mapping) -> Output POSITIVE
                        if error < -AGGRESSIVE_LIMIT:
                            x_val = HARD_TURN_VAL  # Big error? FULL force.
                        else:
                            x_val = SOFT_TURN_VAL  # Small error? Gentle force.
                            
                    else: # error > 0
                        # Target is "Right" of Actual -> Output NEGATIVE
                        if error > AGGRESSIVE_LIMIT:
                            x_val = -HARD_TURN_VAL
                        else:
                            x_val = -SOFT_TURN_VAL

                    self.gamepad.left_joystick_float(x_value_float=x_val, y_value_float=0.0)
                    self.gamepad.update()
                    
                    current_input = x_val # Update for display/logging
            else:
                # >>> MANUAL IDLE <<<
                status = "OFF"
                if current_input != 0:
                    self.gamepad.reset(); self.gamepad.update(); current_input = 0

            # --- 3. VISUALIZATION ---
            if display_img_gray:
                self.update_display(display_img_gray, target_steer, data['steer'], current_input, status)
            
            # Terminal Log
            print(f"[{status}] AI:{target_steer:+.2f} | Real:{data['steer']:+.2f}", end='\r')

        listener.stop()
        pygame.quit()
        print("\nExited.")

if __name__ == "__main__":
    InferenceDriver().run()
    