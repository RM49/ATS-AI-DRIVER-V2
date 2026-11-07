import os
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR = r"data"
LOG_PATH = os.path.join(DATA_DIR, "driving_log.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_PATH = os.path.join(DATA_DIR, "driving_log_cleaned.csv")

def clean_dataset():
    if not os.path.exists(LOG_PATH):
        print(f"Error: Could not find CSV at {LOG_PATH}")
        return

    # Load the log
    print("Loading CSV...")
    df = pd.read_csv(LOG_PATH)
    initial_count = len(df)

    print("Checking for missing images. This may take a moment...")

    # Define a helper to check if the file exists on disk
    def image_exists(filename):
        if pd.isna(filename):
            return False
        return os.path.isfile(os.path.join(IMG_DIR, str(filename)))

    # Apply the check
    # We check 'image_filename' as that is the source of the history columns
    mask = df['image_filename'].apply(image_exists)

    # Filter the dataframe
    df_cleaned = df[mask].copy()
    
    removed_count = initial_count - len(df_cleaned)

    # Save the cleaned file
    df_cleaned.to_csv(OUTPUT_PATH, index=False)

    print("-" * 30)
    print(f"Cleaning Complete!")
    print(f"Total entries processed: {initial_count}")
    print(f"Entries removed:         {removed_count}")
    print(f"Cleaned CSV saved to:    {OUTPUT_PATH}")
    print("-" * 30)
    print("Note: Update your train.py to use 'driving_log_cleaned.csv' now.")

if __name__ == "__main__":
    clean_dataset()