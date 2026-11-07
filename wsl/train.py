import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import Huber

# --- CONFIGURATION ---
DATA_DIR = r"data"
LOG_PATH = os.path.join(DATA_DIR, "driving_log.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")
BATCH_SIZE = 32
EPOCHS = 10

# --- 1. DATA GENERATOR ---
# This function yields batches of data so we don't load 10,000 images into RAM at once.
def batch_generator(data_df, img_folder, batch_size, is_training):
    """
    Yields: ([image_batch, speed_batch], steering_batch)
    """
    data = data_df.values
    # Get column indices for our new columns
    # Note: Check your CSV column order if this crashes, but usually:
    # timestamp(0), center(1), steering(2), throttle(3), speed(4), img_t0(5), img_t1(6), img_t2(7)
    
    # An easier way is to just convert the specific columns we need to numpy
    image_data = data_df[['img_t2', 'img_t1', 'img_t0']].values
    control_data = data_df[['steering', 'speed']].values
    
    num_samples = len(data_df)
    indices = np.arange(num_samples)
    
    while True: 
        np.random.shuffle(indices) # Shuffle is now perfectly safe!
            
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            
            images_batch = []
            speeds_batch = []
            steerings_batch = []
            
            for idx in batch_indices:
                # 1. Load the 3 filenames from our pre-calculated columns
                # row_imgs is [filename_t2, filename_t1, filename_t0]
                row_imgs = image_data[idx] 
                
                # 2. Load Control Data
                steering_angle = float(control_data[idx][0])
                speed = float(control_data[idx][1]) / 35.0 # Normalize Speed!
                
                # 3. Load and Stack Images
                frames = []
                for filename in row_imgs:
                    img_path = os.path.join(img_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        img = np.zeros((66, 200), dtype=np.uint8)
                    
                    frames.append(img)
                
                # Stack along depth: (66, 200, 3)
                # We don't need to reverse anymore because we selected columns in order [t2, t1, t0]
                stacked_img = np.stack(frames, axis=-1)
                
                # Normalize
                stacked_img = stacked_img.astype(np.float32) / 255.0
                
                # Augmentation (Flip)
                # Note: We can't easily check 'is_training' here without passing it in.
                # It is fine to augment everything or pass the flag back in.
                if is_training and np.random.rand() < 0.5:
                    stacked_img = np.fliplr(stacked_img)
                    steering_angle = -steering_angle
                
                images_batch.append(stacked_img)
                speeds_batch.append(speed)
                steerings_batch.append(steering_angle)
            
            yield (
                # Inputs: Dictionary matching Input layer names
                {
                    'vision_input': np.array(images_batch),
                    'speed_input': np.array(speeds_batch)
                }, 
                # Output: Target steering angles
                np.array(steerings_batch)
            )

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
    # output = Dense(1, activation='tanh', name='steering_output')(x)
    output = Dense(1, activation='linear', name='steering_output')(x)

    model = Model(inputs=[img_input, speed_input], outputs=output)
    huber_loss = Huber(delta=0.01)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=huber_loss)
    return model

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load Log
    print("Loading CSV...")
    df = pd.read_csv(LOG_PATH)
    
    # 1. SORT BY TIME (Crucial!)
    # We must ensure data is in order before we calculate history
    df = df.sort_values(by='timestamp')
    
    # 2. CREATE HISTORY COLUMNS (The Fix)
    # We create new columns that hold the filenames of the past frames
    df['img_t0'] = df['image_filename'] # Current
    df['img_t1'] = df['image_filename'].shift(1) # Previous (T-0.1s)
    df['img_t2'] = df['image_filename'].shift(2) # 2nd Previous (T-0.2s)

    # Calculate time difference between current row and previous row
    df['dt'] = df['timestamp'].diff()
    
    df = df[df['speed'] > 1.0]

    # If the gap is larger than 0.2s, it means a new session started.
    # We cannot use history from a previous session.
    mask = df['dt'] > 0.2 

    # Set history to NaN where the gap is too big, so dropna() removes them
    df.loc[mask, 'img_t1'] = np.nan
    df.loc[mask, 'img_t2'] = np.nan

    # df = df.dropna()
    
    # # 3. CLEANUP
    # # The first 2 rows will have NaN because they don't have history. Drop them.
    # df = df.dropna()
    
    # # Optional: Filter stationary data
    # # df = df[df['speed'] > 1.0] 
    
    # print(f"Total samples after processing: {len(df)}")
    
    # 3. CLEANUP & BALANCING
    df = df.dropna()
    df = df[df['speed'] > 1.0] 

    # --- BALANCING THE DATASET ---
    # We want to keep ALL turns, but only a fraction of straight driving.
    # Adjust this threshold based on your data. 0.01 is common for highway trucks.
    STRAIGHT_THRESHOLD = 0.005
    KEEP_PROBABILITY = 0.275 # Keep only 27.5% of straight driving data
    
    # 1. Split into turns and straights
    # We take absolute value because -0.05 (left) is just as important as 0.05 (right)
    df_turns = df[abs(df['steering']) > STRAIGHT_THRESHOLD]
    df_straight = df[abs(df['steering']) <= STRAIGHT_THRESHOLD]
    
    # 2. Randomly sample the straight data
    print(f"Original Straight count: {len(df_straight)}")
    df_straight = df_straight.sample(frac=KEEP_PROBABILITY)
    print(f"Reduced Straight count: {len(df_straight)}")
    
    # 3. Recombine and shuffle
    df = pd.concat([df_turns, df_straight])
    
    # Optional: Visual check (requires matplotlib, helps you verify balance)
    # import matplotlib.pyplot as plt
    # plt.hist(df['steering'], bins=50)
    # plt.show()
    # -----------------------------

    print(f"Total samples after processing: {len(df)}")
    
    # 4. SPLIT (Now safe to shuffle!)
    # Because 'img_t1' is now a column in the row, we can shuffle freely.
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True) 
    
    # Create Generators
    train_gen = batch_generator(train_df, IMG_DIR, BATCH_SIZE, True)
    val_gen = batch_generator(val_df, IMG_DIR, BATCH_SIZE, False)
    
    # Build Model
    model = create_pilotnet_dual_branch()
    model.summary()
    
    # Checkpoints
    checkpoint = ModelCheckpoint(
            'model_best.h5', 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_df) // BATCH_SIZE,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training complete.")
