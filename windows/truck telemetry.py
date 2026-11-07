import mmap
import struct
import time

# configuration matches C++
SHM_NAME = "Local\\SCS_Telemetry_Shared_Mem"
# Struct format: I = unsigned int (4b), f = float (4b)
# We have 1 uint + 5 floats = "Ifffff"
STRUCT_FMT = "Ifffff" 
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)

def main():
    print(f"Attempting to connect to Shared Memory '{SHM_NAME}'...")
    
    shm = None
    while shm is None:
        try:
            # -1 for fileno means "Mapping backed by system paging file" (RAM)
            shm = mmap.mmap(-1, STRUCT_SIZE, tagname=SHM_NAME)
            print("Connected!")
        except FileNotFoundError:
            print("Shared memory not found. Is the game running?")
            time.sleep(1)

    try:
        while True:
            # 1. Move pointer to start of memory block
            shm.seek(0)
            
            # 2. Read the raw bytes
            bytes_data = shm.read(STRUCT_SIZE)
            
            # 3. Unpack bytes into python variables
            timestamp, speed, rpm, steering, throttle, brake = struct.unpack(STRUCT_FMT, bytes_data)
            # Steering -1 is right and +1 is left, industry standards.

            # Print (Use \r to overwrite line for a cleaner UI)
            print(f"Time: {timestamp} | Speed: {speed:.1f} m/s | RPM: {rpm:.0f} | Steer: {steering:.2f}", end="\r")
            
            # Update 20 times per second
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        shm.close()

if __name__ == "__main__":
    main()
