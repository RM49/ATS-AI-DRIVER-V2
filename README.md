# ATS-AI-DRIVER-V2

An imitation-learning driver for American Truck Simulator. A neural network watches the road and steers the truck, while the game's cruise control handles speed. With cruise control on, it can drive down a highway for quite a while.

Version 1: https://github.com/RM49/ATS-AI-Driver

Like V1, this code was AI-written, with Gemini.

## What changed from V1

- **Real telemetry:** a C++ plugin writes telemetry from the game to shared memory, which replaces guessing steering values from key presses.
- **Virtual controller:** steering goes through a virtual gamepad, which allows finer input than key presses and avoids competing for the keyboard.
- **Faster control than prediction:** the control loop runs at 60 Hz and the model at about 10 Hz, so steering can adjust between predictions.
- **Consistent frame timing:** frames are captured about 0.1s apart, in both collection and inference, so the model can learn from how the view changes.
- **Steering only:** the model no longer predicts speed.

## How it works

1. **Plugin:** `plugin/telemetry.cpp` is the SCS SDK's telemetry example, extended to publish speed, RPM and effective steering to a named shared memory block.
2. **Collection:** Python captures the game window, crops it, converts it to greyscale at 200x66, and logs each frame with the current telemetry.
3. **Training:** a PilotNet-style CNN takes a stack of 3 frames plus speed, and predicts steering. Straight-line driving is downsampled so that turns aren't drowned out, and frames are randomly flipped. The dataset is about 105,000 frames before that downsampling, and it is not included here.
4. **Driving:** the model predicts a target steering value, and the control loop pushes the virtual joystick towards it.
5. **Intervention:** holding `a` or `d` while the AI drives hands control back to you and saves those frames, so the model can learn to recover.

## Layout

| Path | Purpose |
|---|---|
| `plugin/` | C++ telemetry plugin source and the SCS SDK licence |
| `windows/` | Data collection, inference and the trained model, run alongside the game |
| `wsl/` | Training scripts, run in WSL |

## Setup

This needs Windows, a copy of American Truck Simulator and the SCS Telemetry SDK (built with version 1.14).

**1. Build the plugin**

Build `plugin/telemetry.cpp` as an x64 DLL in Visual Studio, adding the SDK's `include` folder to the include directories. Then copy the DLL to:

```
<game folder>/bin/win_x64/plugins/
```

**2. Install the Windows dependencies**

```
pip install -r requirements-windows.txt
```

`vgamepad` also needs the ViGEmBus driver.

**3. Drive**

Start the game, then run this from the `windows` folder, using cruise control to set the speed:

```
python InferenceInput.py
```

Press `c` to toggle the AI, hold `a` or `d` to intervene, and press `q` to quit.

## Collecting data and training

Run these from the `windows` folder to collect data, pressing `c` to start and stop recording. Frames are saved to `wsl/data`.

```
python CollectView.py
```

Then, from the `wsl` folder, train:

```
pip install -r ../requirements-training.txt
python train.py
```

Training saves `model_best.h5` to the `wsl` folder. Copy it to `windows` to use it.

## Other scripts

- `windows/truck telemetry.py` prints live telemetry, to check that the plugin works.
- `windows/inference.py` prints predictions next to your own steering, without controlling the truck.
- `windows/InferenceVisual.py` draws the predicted steering over the model's view.
- `windows/Truck collect.py` is an earlier collection script with a different crop. Use `CollectView.py` to match the model.
- `wsl/clean_data.py` removes log rows whose images are missing.

## Known limitations

- The plugin doesn't fill in the shared memory block's throttle and brake fields, so they read as zero.
- The model only steers, so speed relies on cruise control.

## Licence notice

`plugin/telemetry.cpp` is based on the SCS SDK, Copyright (C) 2016 SCS Software. See `plugin/SCS_SDK_LICENSE.txt`.
