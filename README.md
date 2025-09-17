# Autoball

Embedded system based on **Raspberry Pi 5** for real-time tracking of a basketball using computer vision and stepper motor control.

## Repository Structure

- **Config/** → Configuration files (model, parameters, etc.)  
- **ControlMPP/** → C++ code for stepper motor control and PID.  
- **TrainModel/** → Scripts for training the ball detection model (PyTorch).  
- **VideoCapture/** → Video capture from USB/CSI camera or file, sending data to PID.  
- **auto_utils/** → Common utilities (logging, helpers).  
- **test/** → Unit and integration test scripts.  
- **install_dependencies.sh** → Dependency installation script.  
- **requirements.txt** → Python library requirements.  
- **setup.py** → Python environment setup.  

---

## Technical Architecture

Autoball consists of **two main subsystems** working in parallel:

1. **Computer Vision (Python + PyTorch + OpenCV)**  
   - Captures frames from a **USB/CSI camera**.  
   - Processes each frame with a **CNN (optimized ResNet50)** that predicts the ball position `(x, y)` in image coordinates.  
   - Computes the horizontal error relative to the image center (`error_x = x_ball - x_center`).  
   - Sends this error through a **UNIX socket (`/tmp/pid_socket`)** to the C++ controller.  

2. **Motor Control (C++ + libgpiod + PID)**  
   - Receives `error_x` from the socket.  
   - A **PID Controller** calculates the required motor speed to reduce the error.  
   - The **StepperMotor driver** (using libgpiod) generates step/direction signals to the physical driver (A4988/DRV8825).  
   - The camera rotates to keep the ball centered.  

---

## Data Flow

```
[ Camera ] → [ VideoCapture.py ] → [ CNN Model ]
                           ↓
                     (x,y position)
                           ↓
             [ UNIX socket /tmp/pid_socket ]
                           ↓
                [ ControlMPP (C++) ]
                           ↓
         [ PID Controller + StepperMotor ]
                           ↓
                    [ Physical Motor ]
```

---

## PID Control Details

- **Input**: pixel error (`x_pred - x_center`).  
- **Output**: desired angular motor speed.  
- **Implemented strategies**:  
  - `std::clamp()` limits speed to avoid overshoot.  
  - Anti-windup: integral term is bounded.  
  - Fine tuning of `Kp`, `Ki`, `Kd` from config file.  

---

## Python ↔ C++ Communication

- **Channel**: UNIX socket (`/tmp/pid_socket`).  
- **Mode**:  
  - Python **sends** error to PID.  
  - C++ **reads** the value in each control cycle.  

This ensures **low latency** and avoids external dependencies.

---

## Logs & Diagnostics

- Python uses `logging` for FPS, predictions, and errors.  
- C++ uses `auto_utils/logger` for **timestamped logs with levels (INFO/ERROR/DEBUG)**.  
- Logs can be redirected to files for traceability.  

---
 
