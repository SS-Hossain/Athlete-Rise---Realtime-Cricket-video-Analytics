import cv2
import numpy as np
import os
import mediapipe as mp
import json
import time # For FPS calculation

# --- Global MediaPipe Initializations ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Helper functions (These are all correct and need no changes) ---
def calculate_angle(a, b, c, image_width, image_height):
    a_px = np.array([a.x * image_width, a.y * image_height])
    b_px = np.array([b.x * image_width, b.y * image_height])
    c_px = np.array([c.x * image_width, c.y * image_height])
    ba = a_px - b_px
    bc = c_px - b_px
    angle = np.degrees(np.arctan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    angle = abs(angle)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_front_elbow_angle(landmarks, image_width, image_height):
    try:
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        if all(lmk.visibility > 0.6 for lmk in [shoulder, elbow, wrist]):
            return calculate_angle(shoulder, elbow, wrist, image_width, image_height)
        else:
            return None
    except (AttributeError, TypeError):
        return None

def get_spine_lean(landmarks, image_width, image_height):
    try:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if not all(lmk.visibility > 0.6 for lmk in [left_hip, right_hip, left_shoulder, right_shoulder]):
            return None
        mid_hip_x = (left_hip.x + right_hip.x) / 2 * image_width
        mid_hip_y = (left_hip.y + right_hip.y) / 2 * image_height
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2 * image_width
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * image_height
        delta_y = mid_hip_y - mid_shoulder_y
        delta_x = mid_hip_x - mid_shoulder_x
        angle_rad = np.arctan2(delta_x, delta_y)
        angle_deg = np.degrees(angle_rad)
        return abs(angle_deg)
    except (AttributeError, TypeError):
        return None

def get_head_over_knee_alignment(landmarks, image_width, image_height):
    try:
        head = landmarks[mp_pose.PoseLandmark.NOSE]
        front_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        if not all(lmk.visibility > 0.6 for lmk in [head, front_knee]):
            return None
        head_px_x = head.x * image_width
        front_knee_px_x = front_knee.x * image_width
        horizontal_distance = abs(head_px_x - front_knee_px_x)
        return horizontal_distance
    except (AttributeError, TypeError):
        return None

def get_front_foot_direction(landmarks, image_width, image_height):
    try:
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        if not all(lmk.visibility > 0.6 for lmk in [ankle, foot_index]):
            return None
        ankle_px = np.array([ankle.x * image_width, ankle.y * image_height])
        foot_index_px = np.array([foot_index.x * image_width, foot_index.y * image_height])
        foot_vec = foot_index_px - ankle_px
        horizontal_vec = np.array([1, 0])
        dot_product = np.dot(foot_vec, horizontal_vec)
        magnitude_foot = np.linalg.norm(foot_vec)
        magnitude_horizontal = np.linalg.norm(horizontal_vec)
        if magnitude_foot == 0:
            return None
        cosine_angle = np.clip(dot_product / (magnitude_foot * magnitude_horizontal), -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except (AttributeError, TypeError):
        return None

def calculate_velocity(prev_landmark, curr_landmark, image_width, image_height, time_delta):
    if prev_landmark is None or curr_landmark is None or time_delta == 0:
        return 0
    prev_pos = np.array([prev_landmark.x * image_width, prev_landmark.y * image_height])
    curr_pos = np.array([curr_landmark.x * image_width, curr_landmark.y * image_height])
    distance = np.linalg.norm(curr_pos - prev_pos)
    velocity = distance / time_delta
    return velocity

def predict_skill_grade(final_scores):
    scores_list = [details['score'] for details in final_scores.values() if isinstance(details, dict)]
    if not scores_list: return "Not Rated"
    average_score = np.mean(scores_list)
    if average_score >= 7.5: return "Advanced"
    elif average_score >= 4.5: return "Intermediate"
    else: return "Beginner"

def evaluate_shot(metrics_by_phase, thresholds):
    # This function is correct and needs no changes
    scores = {
        "footwork": {"score": 0, "feedback": []}, "head_position": {"score": 0, "feedback": []},
        "swing_control": {"score": 0, "feedback": []}, "balance": {"score": 0, "feedback": []},
        "follow_through": {"score": 0, "feedback": []}
    }
    downswing_metrics = metrics_by_phase.get("DOWNSWING", {})
    if not downswing_metrics or not downswing_metrics.get('elbow_angles'):
        scores['swing_control']['feedback'].append("Could not detect a clear downswing phase for analysis.")
        scores['skill_grade'] = "Not Rated"
        return scores
    impact_frame_index = metrics_by_phase.get("impact_frame_index", -1)
    if impact_frame_index != -1 and impact_frame_index < len(downswing_metrics['head_knee_distances']):
        head_knee_at_impact = downswing_metrics['head_knee_distances'][impact_frame_index]
        foot_angle_at_impact = downswing_metrics['foot_directions'][impact_frame_index]
        if head_knee_at_impact < thresholds['head_knee_dist_good_max'] / 2:
            scores['head_position']['score'] = 9
            scores['head_position']['feedback'].append("Excellent head position at impact, perfectly over the front knee.")
        elif head_knee_at_impact < thresholds['head_knee_dist_good_max']:
            scores['head_position']['score'] = 6
            scores['head_position']['feedback'].append("Good head position over the front knee at impact.")
        else:
            scores['head_position']['score'] = 2
            scores['head_position']['feedback'].append("Work on keeping your head over the front knee at the moment of impact.")
        footwork_score = scores['head_position']['score'] / 2
        if thresholds['foot_angle_good_min'] <= foot_angle_at_impact <= thresholds['foot_angle_good_max']:
            footwork_score += 4
            scores['footwork']['feedback'].append("Front foot is well-directed towards the shot.")
        else:
            scores['footwork']['feedback'].append("Front foot angle could be adjusted for better shot direction.")
        scores['footwork']['score'] = min(10, int(footwork_score))
    avg_elbow_angle = np.mean(downswing_metrics['elbow_angles'])
    swing_control_score = 0
    if thresholds['elbow_angle_good_min'] <= avg_elbow_angle <= thresholds['elbow_angle_good_max']:
        swing_control_score += 6
        scores['swing_control']['feedback'].append("Good elbow elevation during the downswing, providing control and power.")
    else:
        swing_control_score += 2
        scores['swing_control']['feedback'].append("Review elbow elevation during downswing for better control.")
    scores['swing_control']['score'] = swing_control_score
    avg_spine_lean = np.mean(downswing_metrics['spine_leans'])
    balance_score = 0
    if avg_spine_lean < thresholds['spine_lean_good_max']:
        balance_score = 8
        scores['balance']['feedback'].append("Excellent balance with an upright spine during the shot.")
    elif avg_spine_lean < thresholds['spine_lean_good_max'] + 10:
        balance_score = 4
        scores['balance']['feedback'].append("Good balance, but watch for a slight lean.")
    else:
        balance_score = 2
        scores['balance']['feedback'].append("Focus on staying more upright during the shot to improve balance.")
    scores['balance']['score'] = balance_score
    follow_through_metrics = metrics_by_phase.get("FOLLOW_THROUGH", {})
    if follow_through_metrics and follow_through_metrics.get('elbow_angles'):
        max_extension = max(follow_through_metrics['elbow_angles'])
        if max_extension > thresholds['follow_through_min_extension']:
            scores['follow_through']['score'] = 8
            scores['follow_through']['feedback'].append("Excellent extension through the follow-through.")
        else:
            scores['follow_through']['score'] = 4
            scores['follow_through']['feedback'].append("Work on achieving full arm extension in the follow-through.")
    else:
        scores['follow_through']['feedback'].append("Could not clearly analyze the follow-through phase.")
    scores['skill_grade'] = predict_skill_grade(scores)
    return scores

def analyze_video(config: dict):
    # Load config parameters
    video_path = config['video_paths']['input']
    output_dir = config['video_paths']['output_dir']
    pose_config = config['pose_estimation']
    thresholds = config['thresholds']
    phase_thresholds = config['phase_detection']
    
    output_video_name = 'annotated_video.mp4'
    output_eval_file = 'evaluation.json'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_per_frame = 1 / fps if fps > 0 else 0

    output_video_path = os.path.join(output_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    current_phase = "STANCE"
    prev_landmarks = None
    metrics_by_phase = {
        "STANCE": {"elbow_angles": [], "spine_leans": [], "head_knee_distances": [], "foot_directions": []},
        "DOWNSWING": {"elbow_angles": [], "spine_leans": [], "head_knee_distances": [], "foot_directions": []},
        "FOLLOW_THROUGH": {"elbow_angles": [], "spine_leans": [], "head_knee_distances": [], "foot_directions": []},
        "impact_frame_index": -1, "max_wrist_velocity": 0
    }
    
    start_time = time.time()
    frame_count = 0

    print("Starting advanced video analysis...")
    with mp_pose.Pose(min_detection_confidence=pose_config['min_detection_confidence'], 
                      min_tracking_confidence=pose_config['min_tracking_confidence']) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # --- Metric calculation ---
                elbow_angle = get_front_elbow_angle(landmarks, width, height)
                spine_lean = get_spine_lean(landmarks, width, height)
                head_knee_dist = get_head_over_knee_alignment(landmarks, width, height)
                foot_direction = get_front_foot_direction(landmarks, width, height)
                
                # --- Phase Segmentation Logic ---
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                prev_left_wrist = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST] if prev_landmarks else None
                wrist_velocity = calculate_velocity(prev_left_wrist, left_wrist, width, height, time_per_frame)
                
                if current_phase == "STANCE" and wrist_velocity > phase_thresholds['wrist_velocity_threshold_swing']:
                    current_phase = "DOWNSWING"
                elif current_phase == "DOWNSWING" and wrist_velocity < phase_thresholds['wrist_velocity_threshold_stance'] and len(metrics_by_phase['DOWNSWING']['elbow_angles']) > 5:
                    current_phase = "FOLLOW_THROUGH"

                if current_phase in metrics_by_phase:
                    if elbow_angle is not None: metrics_by_phase[current_phase]['elbow_angles'].append(elbow_angle)
                    if spine_lean is not None: metrics_by_phase[current_phase]['spine_leans'].append(spine_lean)
                    if head_knee_dist is not None: metrics_by_phase[current_phase]['head_knee_distances'].append(head_knee_dist)
                    if foot_direction is not None: metrics_by_phase[current_phase]['foot_directions'].append(foot_direction)

                if current_phase == "DOWNSWING":
                    if wrist_velocity > metrics_by_phase['max_wrist_velocity']:
                        metrics_by_phase['max_wrist_velocity'] = wrist_velocity
                        metrics_by_phase['impact_frame_index'] = len(metrics_by_phase['DOWNSWING']['elbow_angles']) - 1

                prev_landmarks = results.pose_landmarks.landmark

                # --- UPGRADED Live Overlays ---
                # Display high-level phase info
                cv2.putText(image_bgr, f"PHASE: {current_phase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Wrist Vel: {wrist_velocity:.1f} px/s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                # Display detailed biomechanical metrics and feedback
                y_offset = 110
                text_color = (0, 255, 0)
                font_scale = 0.7
                font_thickness = 2

                if elbow_angle is not None:
                    text = f"Elbow: {elbow_angle:.1f} deg"
                    cv2.putText(image_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_offset += 25
                    if thresholds['elbow_angle_good_min'] <= elbow_angle <= thresholds['elbow_angle_good_max']:
                        cv2.putText(image_bgr, "[GOOD] Elbow elevation", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(image_bgr, "[FIX] Adjust elbow angle", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), font_thickness, cv2.LINE_AA)
                    y_offset += 30

                if spine_lean is not None:
                    text = f"Spine Lean: {spine_lean:.1f} deg"
                    cv2.putText(image_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_offset += 25
                    if spine_lean < thresholds['spine_lean_good_max']:
                        cv2.putText(image_bgr, "[GOOD] Upright Spine", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(image_bgr, "[FIX] Excessive spine lean", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
                    y_offset += 30

                if head_knee_dist is not None:
                    text = f"Head-Knee Dist: {head_knee_dist:.1f} px"
                    cv2.putText(image_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    y_offset += 25
                    if head_knee_dist < thresholds['head_knee_dist_good_max']:
                         cv2.putText(image_bgr, "[GOOD] Head over front knee", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(image_bgr, "[FIX] Head not over front knee", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
                    y_offset += 30
                
                if foot_direction is not None:
                    text = f"Foot Dir: {foot_direction:.1f} deg"
                    cv2.putText(image_bgr, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)


            out.write(image_bgr)
    
    end_time = time.time()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0
    
    print(f"\nVideo processing complete.")
    print(f"Output saved to: {output_video_path}")
    print(f"Total time: {processing_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    final_evaluation = evaluate_shot(metrics_by_phase, thresholds)
    eval_output_path = os.path.join(output_dir, output_eval_file)
    with open(eval_output_path, 'w') as f:
        json.dump(final_evaluation, f, indent=4)
    print(f"Shot evaluation saved to: {eval_output_path}")

if __name__ == '__main__':
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create it.")
        exit()

    os.makedirs(config['video_paths']['output_dir'], exist_ok=True)
    
    analyze_video(config)