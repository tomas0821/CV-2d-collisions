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

```markdown
## Usage

### Select Ball Type:
* When tracking is not active, press **'1'** to cycle the type for **Ball 1** and **'2'** for **Ball 2**.  
  The current selection is displayed in the bottom-left corner.

### Start Tracking:
* Press **'S'** to start tracking.  
  The application uses YOLO to detect and track the balls, drawing trajectories and velocity vectors.

### Interact with Velocity Vectors:
* Click anywhere on a velocity vector to select it.
* The selected vector will be highlighted in **yellow**, and its velocity (in **cm/s**) will be displayed at the bottom-right corner.

### Toggle Displays:
* Press **'T'** to toggle the trajectory display.
* Press **'V'** to toggle the velocity vector display.

### Stop Tracking:
* Press **'Y'** to stop tracking.

### Reset:
* Press **'R'** to reset all data (trajectories, velocity vectors, and ball type selections).

### Quit:
* Press **'Q'** to quit the application.

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