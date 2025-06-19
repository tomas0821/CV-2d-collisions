import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
from collections import deque

# ---------------- Configuration ----------------
SCALE_LENGTH_CM = 10.0
MODEL_PATH = "pingpong_11n.pt" # A model trained to detect the balls
CAMERA_INDEX = 0
TARGET_WIDTH, TARGET_HEIGHT = 640, 480
TARGET_FPS = 60
CONFIDENCE_THRESHOLD = 0.5
# --- Collision Detection Config ---
DIRECTION_CHANGE_THRESHOLD_DEGREES = 75.0 # Angle to detect a bounce
POINT_HISTORY_LENGTH = 30 
MATCHING_MAX_DISTANCE_PX = 150
MIN_FRAMES_FOR_VALID_TRAJECTORY = 5 # Grace period to prevent false positives at start
COLLISION_DISTANCE_FACTOR = 1.5 # How close balls must be to collide (1.5 = 150% of their combined radii)
VECTOR_SCALE = 2 # Multiplier to make velocity vectors more visible

# --- State Variables ---
calibration_points = []
cm_per_pixel = None
tracking_active = False

# --- Ball 1 State ---
segment_start_time_1, segment_start_pos_1 = None, None
segment_last_time_1, segment_last_pos_1 = None, None
point_history_1 = deque(maxlen=POINT_HISTORY_LENGTH)
frames_tracked_1 = 0 
last_known_radius_1 = 5 

# --- Ball 2 State ---
segment_start_time_2, segment_start_pos_2 = None, None
segment_last_time_2, segment_last_pos_2 = None, None
point_history_2 = deque(maxlen=POINT_HISTORY_LENGTH)
frames_tracked_2 = 0
last_known_radius_2 = 5 

# --- Collision State ---
collision_event_data = {} 
processing_collision = False 
last_reported_collision_data = {} # To store data for persistent vector drawing

# --- General ---
last_time = time.time()

# ---------------- Mouse Callback for Calibration ----------------
def onMouse(event, x, y, flags, param):
    """Handles mouse clicks for setting calibration points."""
    global calibration_points, cm_per_pixel
    if cm_per_pixel is None and event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            if len(calibration_points) == 2:
                pixel_dist = np.linalg.norm(np.array(calibration_points[0]) - np.array(calibration_points[1]))
                if pixel_dist > 0:
                    cm_per_pixel = SCALE_LENGTH_CM / pixel_dist
                    print("-" * 30, f"\nCalibration Complete!\nScale: {cm_per_pixel:.4f} cm/pixel")
                    print("Press 'S' to start tracking or 'R' to reset.\n" + "-" * 30)
                else:
                    print("Error: Points are too close. Resetting.")
                    calibration_points.clear()

# ---------------- Utility Functions ----------------
def calculate_angle(p1, p2, p3):
    """Calculates the angle (in degrees) formed by the path p1 -> p2 -> p3."""
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0: return 0.0
    cos_theta = np.dot(v1, v2) / (mag1 * mag2)
    return math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    if p1 is None or p2 is None: return float('inf')
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_velocity(start_time, end_time, start_pos, end_pos, scale_factor):
    """Calculates speed and velocity vector for a segment."""
    if any(v is None for v in [start_time, end_time, start_pos, end_pos, scale_factor]):
        return None
    
    elapsed_time = end_time - start_time
    if elapsed_time <= 1e-6:
        return None

    pixel_dist_vec = np.array(end_pos) - np.array(start_pos)
    dist_cm = np.linalg.norm(pixel_dist_vec) * scale_factor
    speed_cm_s = dist_cm / elapsed_time
    
    velocity_vector_px_s = (pixel_dist_vec / elapsed_time)
    velocity_vector_cm_s = (velocity_vector_px_s * scale_factor).tolist()

    return {
        "speed_cm_s": speed_cm_s, 
        "vector_px_s": velocity_vector_px_s.tolist(), 
        "vector_cm_s": velocity_vector_cm_s,
        "pos": end_pos
    }

def report_collision(data):
    """Prints a formatted report of the collision dynamics."""
    global last_reported_collision_data
    print("\n" + "="*50)
    print("           COLLISION ANALYSIS REPORT")
    print("="*50)
    
    pre_col_v1_data = data.get('pre_col_v1') or {}
    post_col_v1_data = data.get('post_col_v1') or {}
    pre_col_v2_data = data.get('pre_col_v2') or {}
    post_col_v2_data = data.get('post_col_v2') or {}

    v_pre1 = pre_col_v1_data.get('speed_cm_s', 0)
    v_post1 = post_col_v1_data.get('speed_cm_s', 0)
    v_pre2 = pre_col_v2_data.get('speed_cm_s', 0)
    v_post2 = post_col_v2_data.get('speed_cm_s', 0)

    vec_pre1_cm = pre_col_v1_data.get('vector_cm_s', [0, 0])
    vec_post1_cm = post_col_v1_data.get('vector_cm_s', [0, 0])
    vec_pre2_cm = pre_col_v2_data.get('vector_cm_s', [0, 0])
    vec_post2_cm = post_col_v2_data.get('vector_cm_s', [0, 0])

    print(f"\n--- BALL 1 ---")
    print(f"  Pre-Collision : {v_pre1:7.2f} cm/s (Vx: {vec_pre1_cm[0]:6.2f}, Vy: {vec_pre1_cm[1]:6.2f})")
    print(f"  Post-Collision: {v_post1:7.2f} cm/s (Vx: {vec_post1_cm[0]:6.2f}, Vy: {vec_post1_cm[1]:6.2f})") 

    print(f"\n--- BALL 2 ---")
    print(f"  Pre-Collision : {v_pre2:7.2f} cm/s (Vx: {vec_pre2_cm[0]:6.2f}, Vy: {vec_pre2_cm[1]:6.2f})")
    print(f"  Post-Collision: {v_post2:7.2f} cm/s (Vx: {vec_post2_cm[0]:6.2f}, Vy: {vec_post2_cm[1]:6.2f})")
    
    # <<<< CORRECTED: Perform vector addition for momentum >>>>
    p_initial_x = vec_pre1_cm[0] + vec_pre2_cm[0]
    p_initial_y = vec_pre1_cm[1] + vec_pre2_cm[1]
    
    p_final_x = vec_post1_cm[0] + vec_post2_cm[0]
    p_final_y = vec_post1_cm[1] + vec_post2_cm[1]
    
    # Kinetic energy is scalar, so this remains the same
    ke_initial = 0.5 * (v_pre1**2 + v_pre2**2)
    ke_final = 0.5 * (v_post1**2 + v_post2**2)
    
    print("\n--- SYSTEM DYNAMICS (assuming equal mass) ---")
    # <<<< CORRECTED: Report vector components of total momentum >>>>
    print(f"  Initial Momentum: (Px: {p_initial_x:6.2f}, Py: {p_initial_y:6.2f})")
    print(f"  Final Momentum:   (Px: {p_final_x:6.2f}, Py: {p_final_y:6.2f})")
    print(f"  Initial Kinetic Energy:        {ke_initial:.2f}")
    print(f"  Final Kinetic Energy:          {ke_final:.2f}")
    print("="*50 + "\n")
    
    last_reported_collision_data = data.copy()

def draw_velocity_vectors(frame, data):
    """Draws velocity vectors from the last reported collision."""
    if not data:
        return
    
    pre_v1 = data.get('pre_col_v1')
    post_v1 = data.get('post_col_v1')
    if pre_v1 and 'vector_px_s' in pre_v1:
        start_point = tuple(map(int, pre_v1['pos']))
        end_point = tuple(map(int, np.array(start_point) + np.array(pre_v1['vector_px_s']) * VECTOR_SCALE))
        cv2.arrowedLine(frame, start_point, end_point, (150, 150, 255), 2, tipLength=0.3)
    if post_v1 and 'vector_px_s' in post_v1:
        start_point = tuple(map(int, pre_v1['pos']))
        end_point = tuple(map(int, np.array(start_point) + np.array(post_v1['vector_px_s']) * VECTOR_SCALE))
        cv2.arrowedLine(frame, start_point, end_point, (0, 0, 200), 2, tipLength=0.3)

    pre_v2 = data.get('pre_col_v2')
    post_v2 = data.get('post_col_v2')
    if pre_v2 and 'vector_px_s' in pre_v2:
        start_point = tuple(map(int, pre_v2['pos']))
        end_point = tuple(map(int, np.array(start_point) + np.array(pre_v2['vector_px_s']) * VECTOR_SCALE))
        cv2.arrowedLine(frame, start_point, end_point, (150, 255, 150), 2, tipLength=0.3)
    if post_v2 and 'vector_px_s' in post_v2:
        start_point = tuple(map(int, pre_v2['pos']))
        end_point = tuple(map(int, np.array(start_point) + np.array(post_v2['vector_px_s']) * VECTOR_SCALE))
        cv2.arrowedLine(frame, start_point, end_point, (0, 200, 0), 2, tipLength=0.3)

def match_detections(detections, last_pos_1, last_pos_2):
    det1, det2 = None, None
    if not detections: return None, None
    if len(detections) > 2:
        detections.sort(key=lambda d: d[2], reverse=True); detections = detections[:2]
    
    if len(detections) == 1:
        d = detections[0]
        pos_d = (d[0], d[1])
        dist_to_1 = distance(pos_d, last_pos_1)
        dist_to_2 = distance(pos_d, last_pos_2)
        if last_pos_1 is None and last_pos_2 is None: det1 = d
        elif dist_to_1 < dist_to_2 and dist_to_1 < MATCHING_MAX_DISTANCE_PX: det1 = d
        elif dist_to_2 <= dist_to_1 and dist_to_2 < MATCHING_MAX_DISTANCE_PX: det2 = d

    elif len(detections) == 2:
        d_a, d_b = detections[0], detections[1]
        pos_a, pos_b = (d_a[0], d_a[1]), (d_b[0], d_b[1])
        if last_pos_1 is None and last_pos_2 is None:
            det1, det2 = (d_a, d_b) if pos_a[0] < pos_b[0] else (d_b, d_a)
        else:
            cost_a1_b2 = distance(pos_a, last_pos_1) + distance(pos_b, last_pos_2)
            cost_a2_b1 = distance(pos_a, last_pos_2) + distance(pos_b, last_pos_1)
            if cost_a1_b2 <= cost_a2_b1:
                if distance(pos_a, last_pos_1) < MATCHING_MAX_DISTANCE_PX: det1 = d_a
                if distance(pos_b, last_pos_2) < MATCHING_MAX_DISTANCE_PX: det2 = d_b
            else:
                if distance(pos_a, last_pos_2) < MATCHING_MAX_DISTANCE_PX: det2 = d_a
                if distance(pos_b, last_pos_1) < MATCHING_MAX_DISTANCE_PX: det1 = d_b
    return det1, det2

# ---------------- Setup ----------------
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO model '{MODEL_PATH}' loaded.")
except Exception as e:
    print(f"ERROR loading model: {e}"); exit()

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened(): print(f"Error: Could not open video source {CAMERA_INDEX}."); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT); cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera configured to: {actual_width}x{actual_height}")
window_name = "Elastic Collision Analyzer"; cv2.namedWindow(window_name); cv2.setMouseCallback(window_name, onMouse)
print("\n--- Calibration Required ---"); print(f"Click the two endpoints of a {SCALE_LENGTH_CM} cm reference object.")

# ---------------- Main Loop ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret: print("Error reading frame."); time.sleep(0.5); continue
        current_time = time.time()
        
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False, classes=[0])
        detections_this_frame = [(int((b.xyxy[0][0]+b.xyxy[0][2])/2), int((b.xyxy[0][1]+b.xyxy[0][3])/2), int(b.xyxy[0][2]-b.xyxy[0][0])) for b in results[0].boxes] if results and results[0].boxes else []
        
        matched_det_1, matched_det_2 = match_detections(detections_this_frame, segment_last_pos_1, segment_last_pos_2)
        
        detected_centers_for_drawing = {}
        collision_detected_this_frame = False
        
        # --- Update Ball 1 ---
        if tracking_active and cm_per_pixel and matched_det_1:
            frames_tracked_1 += 1
            center_x, center_y, width = matched_det_1
            current_pos_1 = (center_x, center_y)
            last_known_radius_1 = width / 2
            detected_centers_for_drawing[1] = (center_x, center_y, width)
            point_history_1.append((current_pos_1, current_time))
            segment_last_pos_1, segment_last_time_1 = current_pos_1, current_time
            if segment_start_pos_1 is None:
                segment_start_pos_1, segment_start_time_1 = current_pos_1, current_time
            
            if len(point_history_1) >= 3 and not processing_collision and frames_tracked_1 > MIN_FRAMES_FOR_VALID_TRAJECTORY:
                p3, p2, p1 = point_history_1[-1][0], point_history_1[-2][0], point_history_1[-3][0]
                angle = calculate_angle(p1, p2, p3)
                if angle > DIRECTION_CHANGE_THRESHOLD_DEGREES:
                    if segment_last_pos_2 is not None and distance(p2, segment_last_pos_2) < (last_known_radius_1 + last_known_radius_2) * COLLISION_DISTANCE_FACTOR:
                        collision_detected_this_frame = True
                        print(f"\n*** Collision Detected on Ball 1 (Turn Angle: {angle:.1f}°, Proximity Met) ***")
        else:
            frames_tracked_1 = 0 

        # --- Update Ball 2 ---
        if tracking_active and cm_per_pixel and matched_det_2:
            frames_tracked_2 += 1
            center_x, center_y, width = matched_det_2
            current_pos_2 = (center_x, center_y)
            last_known_radius_2 = width / 2
            detected_centers_for_drawing[2] = (center_x, center_y, width)
            point_history_2.append((current_pos_2, current_time))
            segment_last_pos_2, segment_last_time_2 = current_pos_2, current_time
            if segment_start_pos_2 is None:
                segment_start_pos_2, segment_start_time_2 = current_pos_2, current_time
            
            if len(point_history_2) >= 3 and not processing_collision and frames_tracked_2 > MIN_FRAMES_FOR_VALID_TRAJECTORY:
                p3, p2, p1 = point_history_2[-1][0], point_history_2[-2][0], point_history_2[-3][0]
                angle = calculate_angle(p1, p2, p3)
                if angle > DIRECTION_CHANGE_THRESHOLD_DEGREES:
                    if segment_last_pos_1 is not None and distance(p2, segment_last_pos_1) < (last_known_radius_1 + last_known_radius_2) * COLLISION_DISTANCE_FACTOR:
                        collision_detected_this_frame = True
                        print(f"\n*** Collision Detected on Ball 2 (Turn Angle: {angle:.1f}°, Proximity Met) ***")
        else:
            frames_tracked_2 = 0
        
        # --- Process Collision Event ---
        if collision_detected_this_frame:
            processing_collision = True
            last_reported_collision_data.clear() 
            
            if len(point_history_1) > 1:
                impact_pos_1, impact_time_1 = point_history_1[-2] 
                collision_event_data['pre_col_v1'] = calculate_velocity(segment_start_time_1, impact_time_1, segment_start_pos_1, impact_pos_1, cm_per_pixel)
                segment_start_pos_1 = impact_pos_1
                segment_start_time_1 = impact_time_1
                point_history_1.clear(); point_history_1.append((segment_last_pos_1, segment_last_time_1))
                frames_tracked_1 = 1

            if len(point_history_2) > 1:
                impact_pos_2, impact_time_2 = point_history_2[-2]
                collision_event_data['pre_col_v2'] = calculate_velocity(segment_start_time_2, impact_time_2, segment_start_pos_2, impact_pos_2, cm_per_pixel)
                segment_start_pos_2 = impact_pos_2
                segment_start_time_2 = impact_time_2
                point_history_2.clear(); point_history_2.append((segment_last_pos_2, segment_last_time_2))
                frames_tracked_2 = 1
            
            print("Pre-collision velocities captured. Tracking post-collision paths.")

        # --- Automatic Reporting and Resetting Logic ---
        if processing_collision:
            stable_path_1 = frames_tracked_1 >= MIN_FRAMES_FOR_VALID_TRAJECTORY
            stable_path_2 = frames_tracked_2 >= MIN_FRAMES_FOR_VALID_TRAJECTORY
            
            if stable_path_1 and stable_path_2:
                print("Post-collision paths stabilized. Finalizing report...")
                
                collision_event_data['post_col_v1'] = calculate_velocity(segment_start_time_1, segment_last_time_1, segment_start_pos_1, segment_last_pos_1, cm_per_pixel)
                collision_event_data['post_col_v2'] = calculate_velocity(segment_start_time_2, segment_last_time_2, segment_start_pos_2, segment_last_pos_2, cm_per_pixel)

                report_collision(collision_event_data)
                
                collision_event_data.clear()
                processing_collision = False
                
                segment_start_pos_1, segment_start_time_1 = segment_last_pos_1, segment_last_time_1
                segment_start_pos_2, segment_start_time_2 = segment_last_pos_2, segment_last_time_2
                frames_tracked_1 = 0 
                frames_tracked_2 = 0

        # --- Drawing and Display ---
        display_frame = frame.copy()
        if len(calibration_points) > 0: cv2.circle(display_frame, calibration_points[0], 5, (0, 255, 255), -1)
        if len(calibration_points) > 1: cv2.circle(display_frame, calibration_points[1], 5, (0, 255, 255), -1); cv2.line(display_frame, calibration_points[0], calibration_points[1], (0, 255, 255), 2)
        
        # Trajectory drawing is disabled for debugging
        # if tracking_active:
        #     if len(point_history_1) > 1:
        #         for i in range(1, len(point_history_1)):
        #             p1 = point_history_1[i-1][0]; p2 = point_history_1[i][0]
        #             cv2.line(display_frame, p1, p2, (0, 0, 255), 2)
        #     if len(point_history_2) > 1:
        #          for i in range(1, len(point_history_2)):
        #             p1 = point_history_2[i-1][0]; p2 = point_history_2[i][0]
        #             cv2.line(display_frame, p1, p2, (0, 255, 0), 2)
                    
        draw_velocity_vectors(display_frame, last_reported_collision_data)
        
        if 1 in detected_centers_for_drawing: x,y,w = detected_centers_for_drawing[1]; cv2.circle(display_frame,(x,y),max(5, int(w/2)),(0,0,255),2); cv2.putText(display_frame,"1",(x+int(w/2),y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        if 2 in detected_centers_for_drawing: x,y,w = detected_centers_for_drawing[2]; cv2.circle(display_frame,(x,y),max(5, int(w/2)),(0,255,0),2); cv2.putText(display_frame,"2",(x+int(w/2),y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        
        status_text, color = ("Tracking ACTIVE", (0, 255, 0)) if tracking_active else ("Press 'S' to Track", (0, 255, 255))
        if cm_per_pixel is None: status_text, color = "Calibrate First", (0, 0, 255)
        if processing_collision: status_text, color = "Processing Collision...", (0, 165, 255)
        cv2.putText(display_frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, "Q: Quit | S: Start/Stop | R: Reset", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window_name, display_frame)

        # --- Key Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('s'):
            if tracking_active and processing_collision:
                print("Tracking stopped during collision processing. Finalizing report...")
                collision_event_data['post_col_v1'] = calculate_velocity(segment_start_time_1, segment_last_time_1, segment_start_pos_1, segment_last_pos_1, cm_per_pixel)
                collision_event_data['post_col_v2'] = calculate_velocity(segment_start_time_2, segment_last_time_2, segment_start_pos_2, segment_last_pos_2, cm_per_pixel)
                report_collision(collision_event_data)
            if key == ord('s'):
                tracking_active = not tracking_active; print(f"\nTracking {'STARTED' if tracking_active else 'STOPPED'}.")
                segment_start_pos_1=None; segment_start_pos_2=None; point_history_1.clear(); point_history_2.clear(); collision_event_data.clear(); processing_collision = False
                frames_tracked_1 = 0; frames_tracked_2 = 0; last_reported_collision_data.clear()
            else: break
        elif key == ord('r'):
            print("Resetting calibration and all tracking data.")
            tracking_active=False; cm_per_pixel=None; calibration_points.clear()
            segment_start_pos_1=None; segment_start_pos_2=None; point_history_1.clear(); point_history_2.clear(); collision_event_data.clear(); processing_collision = False
            frames_tracked_1 = 0; frames_tracked_2 = 0; last_reported_collision_data.clear()

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
