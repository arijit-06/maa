# STTPIoT — MAA Project

This repository contains the STTPIoT project (MAA). It includes code for a small hand-gesture / behavior recognition project with Python scripts, ESP32 web files, and an Arduino/MAA firmware folder. The project also includes model files and training artifacts used for behavior recognition.

> Short summary: Gesture detection with MediaPipe + OpenCV, local display and optional device control, plus an Arduino/ESP32 firmware and TensorFlow Lite model artifacts for deployment.

## Repository structure

- `code.py` — original control/experiment script (legacy)
- `script.py` — improved hand-gesture controller (uses MediaPipe, OpenCV)
- `maa/` — Arduino/MAA firmware and header containing model data (`model_data.h`) and related sketches.
- `maa_dev/` — development artifacts for model training and conversion (model files, training data, scripts)
- `esp32/` — ESP32 Web server files and sketch
- `one/` — additional model header(s) (e.g. `model_data.h`) and assets
- `.vscode/` — editor settings (optional)

> Many model files and large binary artifacts are present (HDF5, TFLite). If you plan to keep the repo small, consider moving large binaries to Releases or using Git LFS.

## Quick start (Python - detection)

Requirements (tested on Windows):
- Python 3.8+ (use 3.10 recommended)
- pip packages:
  - opencv-python
  - mediapipe

Optional (if you re-enable network/control code):
  - requests

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe
# requests only if you enable network control in scripts
python -m pip install requests
```

Run the main detection script (shows camera feed and concise console output on finger changes):

```powershell
python "c:\Users\ARIJIT\Desktop\imp\STTPIoT\script.py"
```

Notes:
- `script.py` runs in offline mode by default and will not contact any network device.
- The script processes the unflipped camera frame for stable handedness detection and flips only for display.
- It prints a concise console message when the detected finger count changes (and when > 0).

## Arduino / MAA

The `maa/` folder contains Arduino sketches and a `model_data.h` header (C-style array) which can be used with TensorFlow Lite for Microcontrollers on supported boards.

To upload the Arduino sketch:
1. Open the relevant `.ino` in Arduino IDE or VSCode with PlatformIO.
2. Select the correct board/port.
3. Upload.

## Models and training

- `maa_dev/` contains training scripts and converted TFLite artifacts.
- Large models are included (HDF5/TFLite). If you need to re-train, see `maa_dev/train_model.py` and `maa_dev/generate_training_data.py`.

## Contributing

If you plan to update the repository:
1. Create a branch for your feature/fix.
2. Commit small focused changes.
3. Push and open a Pull Request on GitHub.

## Notes & suggestions

- Consider using Git LFS for `*.h5`, `*.tflite`, and other large binaries.
- Consider moving large training datasets to GitHub Releases or a cloud storage bucket.

## License

Add your preferred license file (e.g. `LICENSE`). If no license exists, the project is not open-source by default.

## Contact

If you need help or want to iterate on README content, tell me which sections you'd like expanded (installation, hardware wiring, model specifics, or a minimal quick demo script) and I will update it.
