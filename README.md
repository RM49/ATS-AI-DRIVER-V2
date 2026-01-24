# ATS-AI-DRIVER-V2

Currently a work in progress

Improvements so far:

- C++ Plugin to write telemetry data from the game to a memory mapped file that the data collection script can access. This eliminates guesswork with telemetry.
- The script now uses a virtual controller for input allowing for more precise input than a keypress and also allows easier keyboard use during test-time as the script is not competing for the keyboard.
- Input frequency by the model is now greater than prediction frequency allowing for rapid adjustments to the steering with consistent time between frames.
- Time between frames captured has been kept consistent to allow for temporal understanding by the model.
- Many changes to the model. It no longer tries to predict speed, speed is handled using in-game cruise control to allow more focus on the steering
