import cv2
from ultralytics import YOLO
import numpy as np
import time

# ---------------- Ball Type Selection ----------------
ball_options = ["Ping Pong", "Golf", "Hollow Golf", "small"]
ball1_type_index = 0
ball2_type_index = 0

# ---------------- Mouse Callback Setup ----------------
click_info = None  # Velocity info string to display at bottom-right
selected_vector_ball = None  # 1 for Ball 1, 2 for Ball 2
selected_vector_index = None  # Index in the corresponding velocity_vectors list
click_threshold = 10  # pixels

def onMouse(event, x, y, flags, param):
    global click_info, selected_vector_ball, selected_vector_index, velocity_vectors_1, velocity_vectors_2
    if event == cv2.EVENT_LBUTTONDOWN:
        # Reset previous selection
        click_info = None
        selected_vector_ball = None
        selected_vector_index = None
        # Check Ball 1's velocity vectors: now each stored as (start, end, (vx,vy), conv_factor)
        for idx, arrow in enumerate(velocity_vectors_1):
            start, end, vel, conv_factor = arrow
            # Use a helper function to compute the minimum distance from click to line segment
            dist = point_to_line_distance((x, y), start, end)
            if dist < click_threshold:
                # Convert velocity to cm/s using the stored conversion factor
                vx_cm = vel[0] * conv_factor
                vy_cm = vel[1] * conv_factor
                click_info = f"Ball 1: vx={vx_cm:.2f}cm/s, vy={vy_cm:.2f}cm/s"
                selected_vector_ball = 1
                selected_vector_index = idx
                break
        if click_info is None:
            for idx, arrow in enumerate(velocity_vectors_2):
                start, end, vel, conv_factor = arrow
                dist = point_to_line_distance((x, y), start, end)
                if dist < click_threshold:
                    vx_cm = vel[0] * conv_factor
                    vy_cm = vel[1] * conv_factor
                    click_info = f"Ball 2: vx={vx_cm:.2f}cm/s, vy={vy_cm:.2f}cm/s"
                    selected_vector_ball = 2
                    selected_vector_index = idx
                    break

def point_to_line_distance(P, A, B):
    """
    Compute the minimum distance from point P to the line segment AB.
    P, A, B are (x, y) tuples.
    """
    P = np.array(P, dtype=float)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    AB = B - A
    if np.linalg.norm(AB) == 0:
        return np.linalg.norm(P - A)
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    t = max(0, min(1, t))
    projection = A + t * AB
    return np.linalg.norm(P - projection)

# ---------------- End Mouse Callback Setup ----------------

# ---------------- Setup ----------------
# For this demo, we load one default model.
model = YOLO("pingpong_11n.pt")
cap = cv2.VideoCapture(0)
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

window_name = "Ball Tracking"
# Make the window resizable
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, onMouse)

# Flags & Controls
tracking_active = False   # 'S' to start tracking, 'Y' to stop
show_trajectory = True    # Toggle with 'T'
show_velocity = True      # Toggle with 'V'

# Previous positions and time for velocity calculations
prev_ball_1 = None   # (x, y) for Ball 1
prev_ball_2 = None   # (x, y) for Ball 2
prev_time = None     # Timestamp

# Lists for storing trajectories and velocity vectors.
trajectory_1 = []         # For Ball 1 (Red)
trajectory_2 = []         # For Ball 2 (Blue)
# Each velocity vector is stored as (start, end, (vx,vy), conv_factor)
velocity_vectors_1 = []
velocity_vectors_2 = []

# Scale factor for velocity vector arrow length
velocity_scale = 0.1

# ---------------- Utility Functions ----------------
def magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    mag1 = magnitude(*v1)
    mag2 = magnitude(*v2)
    if mag1 * mag2 == 0:
        return 0
    cos_theta = np.clip(dot / (mag1 * mag2), -1, 1)
    return np.degrees(np.arccos(cos_theta))

def match_balls(detections, prev_ball_1, prev_ball_2):
    if len(detections) < 2:
        return None, None
    sorted_detections = sorted(detections, key=lambda b: b[0])
    return sorted_detections[0], sorted_detections[1]

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    
    # Display instructions
    cv2.putText(frame, "S: Start | Y: Stop | R: Reset | T: Toggle Trajectory | V: Toggle Velocity | 1: Cycle Ball1 Type | 2: Cycle Ball2 Type | Q: Quit | Click vector for info",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    # When not tracking, display ball type selections in bottom-left
    if not tracking_active:
        cv2.putText(frame, f"Ball 1 Type: {ball_options[ball1_type_index]}", (30, FRAME_HEIGHT - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"Ball 2 Type: {ball_options[ball2_type_index]}", (30, FRAME_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    # Run YOLO detection
    results = model(frame, conf=0.5)
    detected_balls = []
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        detected_balls.append((center_x, center_y, width))
    
    ball_1, ball_2 = match_balls(detected_balls, prev_ball_1, prev_ball_2)
    
    current_time = time.time()
    delta_t = current_time - prev_time if prev_time else 1e-6
    prev_time = current_time
    
    # Process Ball 1
    if ball_1:
        x1, y1, width1 = ball_1
        # Determine physical diameter (in cm) based on ball type selection
        if ball_options[ball1_type_index] in ["Ping Pong", "Golf", "Hollow Golf"]:
            diameter_1 = 4.0
        else:
            diameter_1 = 2.5
        conv_factor_1 = diameter_1 / width1  # cm per pixel conversion factor
        
        if tracking_active:
            trajectory_1.append((x1, y1))
            if prev_ball_1 is not None:
                vx_1 = (x1 - prev_ball_1[0]) / delta_t
                vy_1 = (y1 - prev_ball_1[1]) / delta_t
                # Store velocity vector along with conversion factor
                velocity_vectors_1.append(((x1, y1),
                                             (x1 + int(vx_1 * velocity_scale),
                                              y1 + int(vy_1 * velocity_scale)),
                                             (vx_1, vy_1),
                                             conv_factor_1))
        prev_ball_1 = (x1, y1)
        cv2.circle(frame, (x1, y1), 10, (0,0,255), -1)
        cv2.putText(frame, "Ball 1", (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    # Process Ball 2
    if ball_2:
        x2, y2, width2 = ball_2
        if ball_options[ball2_type_index] in ["Ping Pong", "Golf", "Hollow Golf"]:
            diameter_2 = 4.0
        else:
            diameter_2 = 2.5
        conv_factor_2 = diameter_2 / width2
        
        if tracking_active:
            trajectory_2.append((x2, y2))
            if prev_ball_2 is not None:
                vx_2 = (x2 - prev_ball_2[0]) / delta_t
                vy_2 = (y2 - prev_ball_2[1]) / delta_t
                velocity_vectors_2.append(((x2, y2),
                                             (x2 + int(vx_2 * velocity_scale),
                                              y2 + int(vy_2 * velocity_scale)),
                                             (vx_2, vy_2),
                                             conv_factor_2))
        prev_ball_2 = (x2, y2)
        cv2.circle(frame, (x2, y2), 10, (255,0,0), -1)
        cv2.putText(frame, "Ball 2", (x2+10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    
    # Draw trajectories if enabled
    if show_trajectory:
        for i in range(1, len(trajectory_1)):
            cv2.line(frame, trajectory_1[i-1], trajectory_1[i], (0,0,255), 2)
        for i in range(1, len(trajectory_2)):
            cv2.line(frame, trajectory_2[i-1], trajectory_2[i], (255,0,0), 2)
    
    # Draw velocity vectors if enabled, highlighting selected one in yellow
    if show_velocity:
        for idx, (start, end, vel, conv_factor) in enumerate(velocity_vectors_1):
            if selected_vector_ball == 1 and selected_vector_index == idx:
                cv2.arrowedLine(frame, start, end, (0,255,255), 2)
            else:
                cv2.arrowedLine(frame, start, end, (0,0,255), 2)
        for idx, (start, end, vel, conv_factor) in enumerate(velocity_vectors_2):
            if selected_vector_ball == 2 and selected_vector_index == idx:
                cv2.arrowedLine(frame, start, end, (0,255,255), 2)
            else:
                cv2.arrowedLine(frame, start, end, (255,0,0), 2)
    
    # Draw click info at bottom-right
    if click_info is not None:
        cv2.putText(frame, click_info, (FRAME_WIDTH - 300, FRAME_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("t"):
        show_trajectory = not show_trajectory
        print(f"Trajectory toggled: {show_trajectory}")
    if key == ord("v"):
        show_velocity = not show_velocity
        print(f"Velocity toggled: {show_velocity}")
    if key == ord("s"):
        tracking_active = True
        print("Tracking started")
    if key == ord("y"):
        tracking_active = False
        print("Tracking stopped")
    if key == ord("r"):
        trajectory_1.clear()
        trajectory_2.clear()
        velocity_vectors_1.clear()
        velocity_vectors_2.clear()
        prev_ball_1, prev_ball_2 = None, None
        click_info = None
        selected_vector_ball = None
        selected_vector_index = None
        print("Reset all data")
    if not tracking_active:
        if key == ord("1"):
            ball1_type_index = (ball1_type_index + 1) % len(ball_options)
            print(f"Ball 1 type: {ball_options[ball1_type_index]}")
        if key == ord("2"):
            ball2_type_index = (ball2_type_index + 1) % len(ball_options)
            print(f"Ball 2 type: {ball_options[ball2_type_index]}")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
