# CV 2d collisions

**CV 2d collisions** is a computer vision project that tracks objects in 2D space using a YOLO model and OpenCV. This tool is designed to analyze collisions by tracking different types of balls (Ping Pong, Golf, Hollow Golf, and Small Ball) and visualizing their trajectories and velocity vectors. The application allows you to select the ball type for each ball before starting a run, toggle the display of trajectories and velocity vectors, and interactively click on a velocity vector to reveal its velocity (converted from pixels/s to cm/s using the known ball diameter).

## Features

- **Multi-Ball Type Support:**  
  Select from four ball types:
  - **Ping Pong:** 4.0 cm diameter
  - **Golf:** 4.0 cm diameter
  - **Hollow Golf:** 4.0 cm diameter
  - **Small Ball:** 2.5 cm diameter  
  Use keys **1** (for Ball 1) and **2** (for Ball 2) to cycle through the available ball types before starting a run. The current selection is displayed in the bottom-left corner.

- **Real-Time Tracking:**  
  Uses a YOLO model to detect and track two balls simultaneously.

- **Trajectory & Velocity Visualization:**  
  - Draws trajectories (as lines) and velocity vectors (as arrows) on the video feed.
  - Toggle the display of trajectories and velocity vectors using **'T'** and **'V'** keys.

- **Interactive Velocity Vector Selection:**  
  - Click anywhere along a velocity vector to select it.
  - The selected vector is highlighted in yellow.
  - Its velocity is displayed in cm/s at the bottom-right corner (conversion is based on the known ball diameter).

- **Resizable Interface:**  
  The display window is resizable, allowing you to adjust the view as needed.

## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/tomas0821/CV-2d-collisions.git
   cd CV-2d-collisions

2. **Intalling dependencies:**

   ```bash
   pip install -r requirements.txt

3. **Running:**
   ```bash
   python collision.py

## How the Code Works

The program uses **YOLO object detection** and **OpenCV** to track two balls in real time, drawing their **trajectories** and **velocity vectors**. The user can interact with the interface to analyze motion data.

### **1. Initialization**
- The webcam is accessed using OpenCV (`cv2.VideoCapture(0)`).
- The YOLO model (`pingpong_11n.pt`) is loaded using the `ultralytics` package.
- The display window is resizable (`cv2.WINDOW_NORMAL`).
- Variables for tracking positions, velocities, and user interactions are initialized.

### **2. Ball Detection & Tracking**
- YOLO detects balls in each frame, returning bounding box coordinates.
- Each detected ball is **assigned a label** ("Ball 1" or "Ball 2") to maintain identity across frames.
- The **center of each ball** is recorded for trajectory tracking.
- The ball’s type (Ping Pong, Golf, Hollow Golf, Small Ball) is selectable via **keys '1' and '2'** before starting.

### **3. Velocity Calculation & Visualization**
- Velocity is computed as **displacement over time** in **pixels per second**.
- Using the ball’s known physical diameter, the code **converts pixels to cm** for real-world velocity.
- Velocity vectors are drawn as **arrows** indicating direction and speed.
- The user can **toggle trajectory (T) and velocity vectors (V)** on or off.

### **4. Interactive Features**
- **Clicking on a velocity vector** highlights it in yellow and displays its velocity.
- The selected ball's velocity remains visible until a new selection is made.
- The user can start (`S`), stop (`Y`), reset (`R`), and quit (`Q`) tracking.

### **5. Key Functionalities & Controls**
| Key  | Function |
|------|----------|
| **S**  | Start tracking |
| **Y**  | Stop tracking |
| **R**  | Reset all data |
| **T**  | Toggle trajectory display |
| **V**  | Toggle velocity vector display |
| **1**  | Cycle through ball types for Ball 1 |
| **2**  | Cycle through ball types for Ball 2 |
| **Q**  | Quit program |

This system enables real-time analysis of motion dynamics in **2D collisions**, making it a useful tool for **physics demonstrations and sports analysis**.



---

## Customization

### Changing Resolution:
* You can increase the window size by modifying the `FRAME_WIDTH` and `FRAME_HEIGHT` variables.  
  The window is created with `cv2.WINDOW_NORMAL` for manual resizing.

### Pixel to Centimeter Conversion:
* The code converts velocities from pixels/s to cm/s using the known ball diameter:
  * **Ping Pong, Golf, Hollow Golf:** **4.0 cm** diameter
  * **Small Ball:** **2.5 cm** diameter

### Mouse Callback:
* Click on any part of a velocity vector to reveal its velocity info at the bottom-right corner.
```