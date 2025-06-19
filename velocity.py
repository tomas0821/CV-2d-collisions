import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque

# ---------------- Configuration ----------------
SCALE_LENGTH_CM = 10.0
MODEL_PATH = "pingpong_11n.pt" # Use a model trained for the objects (e.g., pingpong_11n.pt or a general one)
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 640, 480
TARGET_FPS = 60
CONFIDENCE_THRESHOLD = 0.5 # Adjust as needed
# --- Segment Tracking Config ---
DIRECTION_CHANGE_THRESHOLD_DEGREES = 90.0
POINT_HISTORY_LENGTH = 4 # Min 3 needed for angle calc
MATCHING_MAX_DISTANCE_PX = 150 # Max distance to consider a match for reappearing ball

# --- State Variables ---
calibration_points = []
cm_per_pixel = None
tracking_active = False

# --- Ball 1 State ---
segment_start_time_1 = None
segment_start_pos_1 = None
segment_last_pos_1 = None
segment_last_time_1 = None
point_history_1 = deque(maxlen=POINT_HISTORY_LENGTH)
current_segment_number_1 = 0

# --- Ball 2 State ---
segment_start_time_2 = None
segment_start_pos_2 = None
segment_last_pos_2 = None
segment_last_time_2 = None
point_history_2 = deque(maxlen=POINT_HISTORY_LENGTH)
current_segment_number_2 = 0

# --- General ---
frame_count = 0
last_time = time.time()

# ---------------- Mouse Callback for Calibration ----------------
def onMouse(event, x, y, flags, param):
    """Handles mouse clicks for setting calibration points."""
    global calibration_points, cm_per_pixel
    if cm_per_pixel is None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(calibration_points) < 2:
                calibration_points.append((x, y))
                if len(calibration_points) == 2:
                    p1 = np.array(calibration_points[0])
                    p2 = np.array(calibration_points[1])
                    pixel_distance = np.linalg.norm(p1 - p2)
                    if pixel_distance > 0:
                        cm_per_pixel = SCALE_LENGTH_CM / pixel_distance
                        print("-" * 30)
                        print(f"Calibration Complete!")
                        print(f"Scale Factor: {cm_per_pixel:.4f} cm/pixel")
                        print("Press 'S' to start tracking or 'R' to reset calibration.")
                        print("-" * 30)
                    else:
                        print("Error: Points are too close. Please try again.")
                        calibration_points.clear()

# ---------------- Utility Functions ----------------
def calculate_angle(p1, p2, p3):
    """Calculates the angle (in degrees) formed by the path p1 -> p2 -> p3."""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 == 0 or mag2 == 0: return 0.0
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
    # Handle potential domain error for acos if cos_theta is slightly outside [-1, 1]
    if abs(cos_theta) > 1.0:
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def report_segment_speed(start_time, end_time, start_pos, end_pos, scale_factor, segment_number, ball_id):
    """Calculates and prints the average speed for a completed segment."""
    if start_time is None or end_time is None or start_pos is None or end_pos is None or scale_factor is None:
        # This can happen if tracking is stopped before a segment properly starts
        return

    total_time = end_time - start_time
    if total_time <= 1e-6:
        # Avoid division by zero
        return

    pixel_distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    total_distance_cm = pixel_distance * scale_factor
    average_speed_cm_s = total_distance_cm / total_time

    print("-" * 30)
    print(f"Ball {ball_id} - Segment {segment_number} Ended.")
    print(f"  Duration:   {total_time:.2f} s")
    print(f"  Distance:   {total_distance_cm:.2f} cm (straight line)")
    print(f"  Avg Speed:  {average_speed_cm_s:.2f} cm/s")
    print("-" * 30)

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))


# --- Ball Matching Logic (Refined) ---
def match_detections(detections, last_pos_1, last_pos_2):
    """
    Matches new detections to existing ball tracks based on proximity.
    Returns two values: the detection for ball 1, and the detection for ball 2.
    Either can be None if no matching detection is found.
    """
    det1, det2 = None, None

    if not detections:
        return None, None

    # Use only the two largest detections if more than two are found
    if len(detections) > 2:
        detections.sort(key=lambda d: d[2] * d[3], reverse=True) # Sort by area
        detections = detections[:2]

    if len(detections) == 1:
        d = detections[0]
        pos_d = (d[0], d[1])
        dist_to_1 = distance(pos_d, last_pos_1)
        dist_to_2 = distance(pos_d, last_pos_2)

        # If it's the very start (no last positions), assign to ball 1
        if last_pos_1 is None and last_pos_2 is None:
             det1 = d
        # Assign to the closer ball if it's within the matching threshold
        elif dist_to_1 < dist_to_2 and dist_to_1 < MATCHING_MAX_DISTANCE_PX:
            det1 = d
        elif dist_to_2 <= dist_to_1 and dist_to_2 < MATCHING_MAX_DISTANCE_PX:
            det2 = d

    elif len(detections) == 2:
        d_a, d_b = detections[0], detections[1]
        pos_a, pos_b = (d_a[0], d_a[1]), (d_b[0], d_b[1])

        # If it's the very start, assign based on x-coordinate (leftmost is ball 1)
        if last_pos_1 is None and last_pos_2 is None:
            if pos_a[0] < pos_b[0]:
                det1, det2 = d_a, d_b
            else:
                det1, det2 = d_b, d_a
        else:
            # Hungarian algorithm simplified for 2x2: Check two possible assignment costs
            cost_a1_b2 = distance(pos_a, last_pos_1) + distance(pos_b, last_pos_2)
            cost_a2_b1 = distance(pos_a, last_pos_2) + distance(pos_b, last_pos_1)

            if cost_a1_b2 <= cost_a2_b1:
                # Tentative assignment: A -> 1, B -> 2
                if distance(pos_a, last_pos_1) < MATCHING_MAX_DISTANCE_PX:
                    det1 = d_a
                if distance(pos_b, last_pos_2) < MATCHING_MAX_DISTANCE_PX:
                    det2 = d_b
            else:
                # Tentative assignment: A -> 2, B -> 1
                if distance(pos_a, last_pos_2) < MATCHING_MAX_DISTANCE_PX:
                    det2 = d_a
                if distance(pos_b, last_pos_1) < MATCHING_MAX_DISTANCE_PX:
                    det1 = d_b
    return det1, det2


# ---------------- Setup ----------------
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load YOLO model at '{MODEL_PATH}'.")
    print(f"Specific error: {e}")
    exit()


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open video source (Index: {CAMERA_INDEX}).")
    exit()

# Attempt to set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera configured to: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")

window_name = "Two Ball Average Speed Tracker (Segmented)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, onMouse)

print("\n--- Calibration Required ---")
print(f"1. Place a {SCALE_LENGTH_CM} cm reference object in the camera's view.")
print("2. Click on the two endpoints of the object.")

# ---------------- Main Loop ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from camera. End of stream?")
            time.sleep(0.5)
            continue # Use continue to try reading next frame

        current_time = time.time()
        # --- Object Detection ---
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, classes=[0]) # Assuming class 0 is the ball
        detections_this_frame = []

        # Extract all valid detections from the current frame
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                detections_this_frame.append((center_x, center_y, width, height))

        # --- Match Detections to Ball Tracks ---
        matched_det_1, matched_det_2 = match_detections(
            detections_this_frame,
            segment_last_pos_1,
            segment_last_pos_2
        )

        # --- Update State for Each Ball ---
        # A dictionary to hold the final positions for drawing this frame
        detected_centers_for_drawing = {}

        # Process Ball 1
        if tracking_active and cm_per_pixel is not None:
            if matched_det_1:
                center_x, center_y, width, _ = matched_det_1
                current_pos_1 = (center_x, center_y)
                detected_centers_for_drawing[1] = (center_x, center_y, width)

                current_point_1 = (current_pos_1[0], current_pos_1[1], current_time)
                point_history_1.append(current_point_1)
                segment_last_pos_1 = current_pos_1
                segment_last_time_1 = current_time

                if segment_start_pos_1 is None:
                    # This is the first time we see this ball in a new tracking session
                    segment_start_pos_1 = current_pos_1
                    segment_start_time_1 = current_time
                    current_segment_number_1 = 1 # Start with segment 1
                    point_history_1.clear()
                    point_history_1.append(current_point_1)
                    print(f"Ball 1 - Segment {current_segment_number_1}: Started at {current_pos_1}")

                elif len(point_history_1) >= 3:
                    p3, p2, p1 = point_history_1[-1], point_history_1[-2], point_history_1[-3]
                    angle = calculate_angle((p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]))

                    if angle > DIRECTION_CHANGE_THRESHOLD_DEGREES:
                        print(f"Ball 1: Sharp turn detected ({angle:.1f}°). Ending segment {current_segment_number_1}.")
                        report_segment_speed(segment_start_time_1, p2[2], segment_start_pos_1, (p2[0], p2[1]), cm_per_pixel, current_segment_number_1, 1)

                        # Start a new segment from the point where the turn occurred (p2)
                        current_segment_number_1 += 1
                        segment_start_pos_1 = (p2[0], p2[1])
                        segment_start_time_1 = p2[2]
                        # The history for the new segment should contain the turn point and the new point
                        point_history_1.clear()
                        point_history_1.append(p2)
                        point_history_1.append(p3)
                        print(f"Ball 1 - Segment {current_segment_number_1}: Started at {segment_start_pos_1}")

        # Process Ball 2
        if tracking_active and cm_per_pixel is not None:
            if matched_det_2:
                center_x, center_y, width, _ = matched_det_2
                current_pos_2 = (center_x, center_y)
                detected_centers_for_drawing[2] = (center_x, center_y, width)

                current_point_2 = (current_pos_2[0], current_pos_2[1], current_time)
                point_history_2.append(current_point_2)
                segment_last_pos_2 = current_pos_2
                segment_last_time_2 = current_time

                if segment_start_pos_2 is None:
                    segment_start_pos_2 = current_pos_2
                    segment_start_time_2 = current_time
                    current_segment_number_2 = 1
                    point_history_2.clear()
                    point_history_2.append(current_point_2)
                    print(f"Ball 2 - Segment {current_segment_number_2}: Started at {current_pos_2}")

                elif len(point_history_2) >= 3:
                    p3, p2, p1 = point_history_2[-1], point_history_2[-2], point_history_2[-3]
                    angle = calculate_angle((p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]))

                    if angle > DIRECTION_CHANGE_THRESHOLD_DEGREES:
                        print(f"Ball 2: Sharp turn detected ({angle:.1f}°). Ending segment {current_segment_number_2}.")
                        report_segment_speed(segment_start_time_2, p2[2], segment_start_pos_2, (p2[0], p2[1]), cm_per_pixel, current_segment_number_2, 2)

                        current_segment_number_2 += 1
                        segment_start_pos_2 = (p2[0], p2[1])
                        segment_start_time_2 = p2[2]
                        point_history_2.clear()
                        point_history_2.append(p2)
                        point_history_2.append(p3)
                        print(f"Ball 2 - Segment {current_segment_number_2}: Started at {segment_start_pos_2}")

        # --- Draw Visuals on a fresh copy of the frame ---
        display_frame = frame.copy()

        # Draw Calibration markers
        if len(calibration_points) > 0: cv2.circle(display_frame, calibration_points[0], 5, (0, 255, 255), -1)
        if len(calibration_points) > 1:
            cv2.circle(display_frame, calibration_points[1], 5, (0, 255, 255), -1)
            cv2.line(display_frame, calibration_points[0], calibration_points[1], (0, 255, 255), 2)

        # Draw Ball 1 Marker & Path
        if 1 in detected_centers_for_drawing:
            x, y, width_px = detected_centers_for_drawing[1]
            radius_draw = max(5, int(width_px / 2))
            color1 = (0, 0, 255) # Red for Ball 1
            cv2.circle(display_frame, (x, y), radius_draw, color1, 2)
            cv2.putText(display_frame, "1", (x + radius_draw, y - radius_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
        if tracking_active and segment_start_pos_1 and segment_last_pos_1:
             cv2.line(display_frame, segment_start_pos_1, segment_last_pos_1, (0, 0, 255), 1)

        # Draw Ball 2 Marker & Path
        if 2 in detected_centers_for_drawing:
            x, y, width_px = detected_centers_for_drawing[2]
            radius_draw = max(5, int(width_px / 2))
            color2 = (0, 255, 0) # Green for Ball 2
            cv2.circle(display_frame, (x, y), radius_draw, color2, 2)
            cv2.putText(display_frame, "2", (x + radius_draw, y - radius_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color2, 2)
        if tracking_active and segment_start_pos_2 and segment_last_pos_2:
             cv2.line(display_frame, segment_start_pos_2, segment_last_pos_2, (0, 255, 0), 1)

        # --- Display Status Text & Timers ---
        fps_text = f"FPS: {1.0 / (current_time - last_time):.1f}"
        cv2.putText(display_frame, fps_text, (10, actual_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        last_time = current_time

        status_text, status_color = "", (255,255,255)
        if cm_per_pixel is None:
            status_text = f"Click {'END' if len(calibration_points)==1 else 'START'} of {SCALE_LENGTH_CM}cm scale"
            status_color = (0, 255, 255) # Yellow
        elif not tracking_active:
            status_text = "Calibrated. Press 'S' to Track."
            status_color = (0, 255, 0) # Green
        else:
            status_text = f"Tracking ACTIVE (Scale: {cm_per_pixel:.3f} cm/px)"
            status_color = (0, 255, 0) # Green
        cv2.putText(display_frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display_frame, "Q: Quit | S: Start/Stop | R: Reset Calib", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display Segment Timers
        if tracking_active and segment_start_time_1:
            timer_1 = f"B1 Seg{current_segment_number_1}: {time.time() - segment_start_time_1:.1f}s"
            cv2.putText(display_frame, timer_1, (actual_width - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if tracking_active and segment_start_time_2:
            timer_2 = f"B2 Seg{current_segment_number_2}: {time.time() - segment_start_time_2:.1f}s"
            cv2.putText(display_frame, timer_2, (actual_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # --- Display Frame ---
        cv2.imshow(window_name, display_frame)

        # --- Key Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if tracking_active: # Report final segments before quitting
                report_segment_speed(segment_start_time_1, segment_last_time_1, segment_start_pos_1, segment_last_pos_1, cm_per_pixel, current_segment_number_1, 1)
                report_segment_speed(segment_start_time_2, segment_last_time_2, segment_start_pos_2, segment_last_pos_2, cm_per_pixel, current_segment_number_2, 2)
            print("Quitting program.")
            break
        elif key == ord('s'):
            if cm_per_pixel is None:
                print("Action Required: Please calibrate the scale first!")
            else:
                tracking_active = not tracking_active
                if tracking_active:
                    print("Tracking STARTED. Waiting for detections...")
                    # Reset all states to ensure a clean start
                    point_history_1.clear(); point_history_2.clear()
                    segment_start_pos_1 = None; segment_start_time_1 = None; segment_last_pos_1 = None; segment_last_time_1 = None; current_segment_number_1 = 0
                    segment_start_pos_2 = None; segment_start_time_2 = None; segment_last_pos_2 = None; segment_last_time_2 = None; current_segment_number_2 = 0
                else:
                    print("Tracking STOPPED.")
                    # Report the final segment for each ball upon stopping
                    report_segment_speed(segment_start_time_1, segment_last_time_1, segment_start_pos_1, segment_last_pos_1, cm_per_pixel, current_segment_number_1, 1)
                    report_segment_speed(segment_start_time_2, segment_last_time_2, segment_start_pos_2, segment_last_pos_2, cm_per_pixel, current_segment_number_2, 2)
        elif key == ord('r'):
            print("Resetting calibration and all tracking data.")
            tracking_active = False
            cm_per_pixel = None
            calibration_points.clear()
            # Reset all state variables
            point_history_1.clear(); point_history_2.clear()
            segment_start_pos_1 = None; segment_start_time_1 = None; segment_last_pos_1 = None; segment_last_time_1 = None; current_segment_number_1 = 0
            segment_start_pos_2 = None; segment_start_time_2 = None; segment_last_pos_2 = None; segment_last_time_2 = None; current_segment_number_2 = 0


except Exception as e:
    print(f"An unexpected error occurred in the main loop: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup resources
    if cap.isOpened():
        cap.release()
        print("Camera released.")
    cv2.destroyAllWindows()
    print("Windows closed.")
