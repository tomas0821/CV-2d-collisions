# 2D Collision Analyzer using OpenCV and YOLO

This Python script uses a standard webcam and a YOLOv8 object detection model to perform a 2D physics analysis of collisions between two identical objects (e.g., ping-pong balls). It tracks the objects in real-time, detects collision events, and calculates pre- and post-collision velocities to analyze momentum and kinetic energy.

## Features

-   **Real-time Tracking:** Tracks two objects simultaneously using their center points.
-   **Scale Calibration:** A simple mouse-click calibration allows you to convert pixel distances to real-world units (cm).
-   **Advanced Collision Detection:** Uses a robust, multi-factor system to detect collisions, requiring both a sharp change in direction and physical proximity.
-   **Physics Analysis:** Automatically generates a command-line report after each collision, detailing:
    -   Pre- and post-collision speeds for each object.
    -   Velocity components (Vx, Vy) for each object.
    -   Total system momentum (vector components Px, Py).
    -   Total system kinetic energy.
-   **Multi-Collision Support:** Automatically resets after a collision is reported, allowing for the analysis of multiple events in a single session.
-   **Visual Feedback:**
    -   Displays object markers and IDs.
    -   Draws velocity vectors (pre- and post-collision) on-screen for intuitive analysis.

## Requirements

You will need Python 3 installed, along with the following libraries:

-   `ultralytics`
-   `opencv-python`
-   `numpy`

You will also need a YOLOv8 model file (`.pt`) trained to detect the objects you want to track. A pre-trained model for ping-pong balls (`pingpong_11n.pt`) is specified in the code.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install Dependencies:**
    ```bash
    pip install ultralytics opencv-python numpy
    ```

3.  **Download YOLO Model:**
    Place your trained YOLOv8 model file (e.g., `pingpong_11n.pt`) in the same directory as the Python script. Ensure the `MODEL_PATH` variable in the script matches the name of your model file.

## How to Run

1.  **Execute the Script:**
    Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```
    A window titled "Elastic Collision Analyzer" will open, showing your webcam feed.

2.  **Calibrate:**
    -   Place an object of a known length (e.g., a 10 cm ruler) in the camera's view.
    -   Click on one end of the object in the window.
    -   Click on the other end.
    -   The terminal will print a "Calibration Complete!" message.

3.  **Track and Analyze:**
    -   Press the `'s'` key to start tracking.
    -   Introduce the two balls into the frame and initiate a collision.
    -   The script will automatically detect the collision, wait for the post-collision paths to stabilize, and print a full analysis report in the terminal.
    -   After a report is printed, the system is ready to detect the next collision.

4.  **Controls:**
    -   `s`: Start/Stop tracking.
    -   `r`: Reset calibration and all tracking data.
    -   `q`: Quit the application.

## Configuration

You can adjust the tracking and detection behavior by modifying the configuration variables at the top of the script:

-   `SCALE_LENGTH_CM`: The length of your reference object for calibration.
-   `MODEL_PATH`: The filename of your YOLO model.
-   `CONFIDENCE_THRESHOLD`: The minimum confidence for a detection to be considered valid (0.0 to 1.0).
-   `DIRECTION_CHANGE_THRESHOLD_DEGREES`: The angle that defines a "sharp turn" for collision detection.
-   `COLLISION_DISTANCE_FACTOR`: A multiplier for how close the objects must be to trigger a collision.
-   `VECTOR_SCALE`: A visual multiplier to make the drawn velocity vectors longer and more visible.
