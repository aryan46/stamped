import sys
import ultralytics
import yt_dlp
import time
print(f"✅ Python Executable: {sys.executable}")
print(f"✅ Ultralytics Version: {ultralytics.__version__}")
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
import threading
import time
import os
import pathlib
import requests
from ultralytics import YOLO
import torch
from collections import deque
import time
# from sklearn.linear_model import LinearRegression # Removed to avoid dependency



import json

# --- Flask App Setup & Secret Key for Sessions ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_for_session' 

# --- Load Venue Configuration (Generic Platform Logic) ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
VIDEO_DIR = BASE_DIR / "videos"
CONFIG_FILE = pathlib.Path(__file__).resolve().parent / "venue_config.json"

def load_venue_config():
    if not CONFIG_FILE.exists():
        print("CRITICAL ERROR: venue_config.json not found!")
        sys.exit(1)
        
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

VENUE_CONFIG = load_venue_config()
VENUE_INFO = VENUE_CONFIG.get("venue_info", {})
DEFAULT_THRESHOLDS = VENUE_CONFIG.get("settings", {}).get("default_thresholds", {})

print(f"✅ Loaded Configuration for Venue: {VENUE_INFO.get('name')} ({VENUE_INFO.get('type')})")

# --- Initialize Data Structures from Config ---
AUTHORITY_USERS = {}
USER_DATA = {}
CAMERA_CONFIGS = {}
GATE_CAMERA_MAPPING = {
    "gate1": "zone_gate1_queue",
    "gate2": "zone_gate2_queue",
    "gate3": "zone_outer_waiting" 
}
for user, details in VENUE_CONFIG.get("users", {}).items():
    AUTHORITY_USERS[user] = details.get("password")
    USER_DATA[user] = { "sources": {}, "labels": {} }
    
    for cam in details.get("cameras", []):
        cam_id = cam["id"]
        source_val = cam["source"]
        
        # Auto-resolve file path if it looks like a file
        if not str(source_val).isdigit() and not str(source_val).startswith("http"):
             potential_path = VIDEO_DIR / source_val
             if potential_path.exists():
                 source_val = str(potential_path)
             else:
                 # Fallback for relative paths or direct strings
                 pass

        USER_DATA[user]["sources"][cam_id] = source_val
        USER_DATA[user]["labels"][cam_id] = cam["label"]
        
        # Load specific configs or defaults
        cam_conf = cam.get("config", {})
        CAMERA_CONFIGS[cam_id] = {
            "MAX_CAPACITY": cam_conf.get("max_capacity", DEFAULT_THRESHOLDS.get("max_capacity", 30)),
            "PROXIMITY_THRESHOLD": cam_conf.get("proximity_threshold", DEFAULT_THRESHOLDS.get("proximity_threshold", 80)),
            "HIGH_RISK_THRESHOLD": cam_conf.get("high_risk", DEFAULT_THRESHOLDS.get("high_risk", 0.6)),
            "PROXIMITY_THRESHOLD": cam_conf.get("proximity_threshold", DEFAULT_THRESHOLDS.get("proximity_threshold", 80)),
            "HIGH_RISK_THRESHOLD": cam_conf.get("high_risk", DEFAULT_THRESHOLDS.get("high_risk", 0.6)),
            "STAMPEDE_THRESHOLD": cam_conf.get("stampede", DEFAULT_THRESHOLDS.get("stampede", 0.85)),
            "AREA_SQ_METERS": cam_conf.get("area_sq_meters", 20.0) # Default 20 sqm if not set
        }

# Global maps for detection threads
ALL_VIDEO_SOURCES = {} 
for user, data in USER_DATA.items():
    ALL_VIDEO_SOURCES.update(data["sources"])

PROCESSING_INTERVAL_SECONDS = VENUE_CONFIG.get("settings", {}).get("processing_interval", 3.0)
DEFAULT_CAMERA_CONFIG = { 
    "MAX_CAPACITY": DEFAULT_THRESHOLDS.get("max_capacity", 30), 
    "PROXIMITY_THRESHOLD": DEFAULT_THRESHOLDS.get("proximity_threshold", 80), 
    "HIGH_RISK_THRESHOLD": DEFAULT_THRESHOLDS.get("high_risk", 0.6), 
    "PROXIMITY_THRESHOLD": DEFAULT_THRESHOLDS.get("proximity_threshold", 80), 
    "HIGH_RISK_THRESHOLD": DEFAULT_THRESHOLDS.get("high_risk", 0.6), 
    "STAMPEDE_THRESHOLD": DEFAULT_THRESHOLDS.get("stampede", 0.85),
    "AREA_SQ_METERS": 20.0 
}

# --- Thread-safe dictionaries ---
output_frames = {}
status_data = {}
locks = {} # Will be populated dynamically
STOP_EVENTS = {} # To control thread termination

# --- PREDICTIVE ANALYTICS ENGINE ---
class CrowdPredictor:
    def __init__(self):
        self.history = {} # { cam_id: deque([(time, count), ...], maxlen=20) }
        self.lock = threading.Lock()

    def update(self, cam_id, count):
        with self.lock:
            if cam_id not in self.history:
                self.history[cam_id] = deque(maxlen=20) # Store last ~60 seconds (3s interval)
            self.history[cam_id].append((time.time(), count))

    def get_prediction(self, cam_id, max_cap):
        with self.lock:
            if cam_id not in self.history or len(self.history[cam_id]) < 5:
                return { "trend": "Stabilizing", "time_to_full": "N/A", "flow_rate": 0 }
            
            data = list(self.history[cam_id])
            # Simple Linear Regression: count = m * time + c
            # We calculate slope 'm' (people per second)
            
            times = np.array([x[0] - data[0][0] for x in data]) # Normalize time to 0 start
            counts = np.array([x[1] for x in data])
            
            if len(times) < 2: return { "trend": "Stabilizing", "time_to_full": "N/A", "flow_rate": 0 }

            # Slope (m) = cov(t, c) / var(t)
            m = np.polyfit(times, counts, 1)[0]
            
            flow_per_min = m * 60
            current_count = counts[-1]
            
            trend = "Stable"
            if flow_per_min > 2: trend = "Rising Rapidly ⬆️"
            elif flow_per_min > 0.5: trend = "Increasing ↗️"
            elif flow_per_min < -2: trend = "Clearing Quickly ⬇️"
            elif flow_per_min < -0.5: trend = "Decreasing ↘️"
            
            time_to_full = "Safe"
            if m > 0.1 and current_count < max_cap:
                # Time = (Max - Current) / Rate
                seconds_left = (max_cap - current_count) / m
                if seconds_left < 300: # Less than 5 mins
                    time_to_full = f"CRITICAL: < {int(seconds_left/60)} mins"
                elif seconds_left < 600:
                    time_to_full = f"Warning: ~{int(seconds_left/60)} mins"
            
            return {
                "trend": trend,
                "time_to_full": time_to_full,
                "flow_rate": round(flow_per_min, 1) # People / Min change
            }

PREDICTOR = CrowdPredictor()

# --- Alert Function (Unchanged) ---

# --- Alert Function (Unchanged) ---
def send_alert(alert_type, location="N/A", details=""):
    url = "https://webhook.site/9a01f8af-fcbc-4f5b-befd-f14c99539a5f"
    data = {"alert_type": alert_type, "location": location, "details": details, "timestamp": time.ctime()}
    try:
        requests.post(url, json=data)
        print(f"ALERT SENT: {alert_type} at {location}. Details: {details}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending alert: {e}")

# --- Detection Logic (Full Original Code Restored) ---
def run_stampede_detection(camera_id, source_path, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[{camera_id}] Starting detection on {source_path}")
    
    # Use specific config if exists, else default
    config = CAMERA_CONFIGS.get(camera_id, DEFAULT_CAMERA_CONFIG)
    MIN_PERSON_COUNT = 3  # Lowered gate to ensure low-count crowds are still analyzed

    # Handle webcam index (int) vs file/rtsp (str)
    if str(source_path).isdigit():
        cap = cv2.VideoCapture(int(source_path))
    else:
        cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"Cannot open video for {camera_id} at {source_path}")
        err_img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(err_img, 'Connection Failed', (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(err_img, 'Check URL / Server', (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if camera_id not in locks: locks[camera_id] = threading.Lock()
        with locks[camera_id]:
            output_frames[camera_id] = err_img
            status_data[camera_id] = {"location": camera_id, "situation": "Connection Error", "person_count": 0, "avg_motion": 0, "risk_score": 0}
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1 / fps if fps > 0 else 0.033

    ret, prev_frame = cap.read()
    if not ret: return
    prev_frame = cv2.resize(prev_frame, (640, 360))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    last_person_count = 0
    last_avg_motion = 0.0
    last_situation = "Initializing..."
    last_risk_score = 0.0
    last_processing_time = time.time()
    
    while True:
        if STOP_EVENTS.get(camera_id) and STOP_EVENTS[camera_id].is_set():
            print(f"Stopping thread for {camera_id}")
            break



        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 360))
        
        if (time.time() - last_processing_time) > PROCESSING_INTERVAL_SECONDS:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            valid_flow = mag > 0.2
            if np.any(valid_flow):
                avg_motion, std_motion = np.mean(mag[valid_flow]), np.std(mag[valid_flow])
            else:
                avg_motion, std_motion = 0.0, 0.0

            # Sensitivity Tuned: Lowered conf to 0.15 to better detect people in blurry/dense crowds
            results = model.predict(frame, conf=0.15, iou=0.5, classes=[0], device=device, verbose=False)
            
            # --- PROXIMITY LOGIC ---
            centroids = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = int((x1+x2)/2), int(y2) # Feet position
                    centroids.append((cx, cy))
            
            person_count = len(centroids)

            # --- VISUAL OCCUPANCY (OPEN SPACE) ---
            # Calculate how much of the screen is covered by people bounding boxes.
            # If screen is < 40% covered, there is "Open Space" -> Reduce Risk.
            screen_area = 640 * 360
            occupied_area = 0
            if results[0].boxes is not None:
                for box in results[0].boxes:
                     w, h = box.xywh[0][2], box.xywh[0][3]
                     occupied_area += float(w * h)
            
            occupancy_ratio = min(occupied_area / screen_area, 1.0)
            open_space_buffer = 1.0
            if occupancy_ratio < 0.40:
                # "Open Space" Logic: If < 40% screen is filled, apply safety buffer
                # 10% filled -> 0.5x risk, 40% filled -> 1.0x risk
                open_space_buffer = 0.5 + (occupancy_ratio / 0.40) * 0.5
            
            # Calculate Clustering (People touching/too close)
            close_pairs = 0
            clustered_people = set()
            
            # Distance Matrix Calculation
            prox_thresh = config.get("PROXIMITY_THRESHOLD", 80)
            if person_count > 1:
                # Naive O(N^2) is fine for N < 100
                ct_arr = np.array(centroids)
                for i in range(person_count):
                    for j in range(i + 1, person_count):
                        dist = np.linalg.norm(ct_arr[i] - ct_arr[j])
                        if dist < prox_thresh:
                            close_pairs += 1
                            clustered_people.add(i)
                            clustered_people.add(j)

            clustered_count = len(clustered_people)
            cluster_ratio = clustered_count / person_count if person_count > 0 else 0
            
            # --- ADVANCED RISK FORMULA ---
            
            
            # --- UPDATED LOGIC: NOISE FILTER & DENSITY ---
            # 0. Noise Filter: Ignore if < 3 people (Likely just camera noise or single person)
            if person_count < 3:
                risk_score = 0.0
                current_situation = "Safe"
                # Still record data, but force safe status
            
            else:
                # 1. Density Calculation (People per Sq Meter)
                # Standard Safety: < 2/m² Safe, 2-4 Crowded, 4-5 High Risk, > 5.5 Stampede
                area = config.get("AREA_SQ_METERS", 20.0)
                density = person_count / area
                
                # Normalize Density Risk (0.0 to 1.0)
                # Map 2.0 -> 0.4 (Crowded start), 4.0 -> 0.75 (High Risk), 5.5 -> 1.0 (Critical)
                density_risk = 0.0
                if density > 5.5: density_risk = 1.0
                elif density > 4.0: density_risk = 0.75 + ((density - 4.0) / 1.5 * 0.25)
                elif density > 2.0: density_risk = 0.40 + ((density - 2.0) / 2.0 * 0.35)
                else: density_risk = density / 5.0 # Low linear risk
                
                # 2. Clustering (Pressure) - Still keep this as it's useful
                cluster_risk = cluster_ratio
                
                # 3. Motion (Panic)
                norm_avg_motion = min(avg_motion / 10.0, 1.0)

                # WEIGHTED RISK SCORE (Updated for Density Priority)
                # 60% Density, 30% Clustering, 10% Motion
                risk_score = (density_risk * 0.60) + (cluster_risk * 0.30) + (norm_avg_motion * 0.10)
                
                # Apply Open Space Buffer
                risk_score *= open_space_buffer
                
                # --- FAILSAFE: CHAOTIC MOTION OVERRIDE ---
                # If motion is high (> 5.0) AND there are people (> 5), it implies panic/chaos
                if avg_motion > 5.0 and person_count > 5:
                    chaos_adder = min((avg_motion - 5.0) * 0.1, 0.4) # Add up to 40% risk
                    risk_score += chaos_adder
                
                risk_score = max(0, min(risk_score, 1.0))

                current_situation = "Safe"
                if risk_score >= config.get("STAMPEDE_THRESHOLD", 0.85):
                    current_situation = "Stampede in Progress"
                elif risk_score >= config.get("HIGH_RISK_THRESHOLD", 0.6):
                    current_situation = "High Risk of Stampede"
                elif risk_score > 0.35: 
                    current_situation = "Crowded"
            
            if current_situation != "Safe" and current_situation != last_situation:
                alert_details = f"People: {person_count}, Density: {person_count/config.get('AREA_SQ_METERS', 20):.2f}/m², Risk: {risk_score:.2f}"
                send_alert(current_situation, location=camera_id.upper(), details=alert_details)
            
            # Update Predictor
            PREDICTOR.update(camera_id, person_count)

            last_person_count, last_avg_motion, last_risk_score, last_situation = person_count, avg_motion, risk_score, current_situation
            prev_gray = gray
            last_processing_time = time.time()
        
        overlay = frame.copy()
        alert_color = (0, 255, 0)
        if "Stampede" in last_situation: alert_color = (0, 0, 255)
        elif "High Risk" in last_situation: alert_color = (0, 165, 255)
        elif "Crowded" in last_situation: alert_color = (0, 255, 255)

        # Draw Header
        cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
        
        # --- DRAW VISUAL PRESSURE LINES ---
        # Draw lines between detected centroids if they are close
        # This gives a 'Web' effect showing density pressure
        if 'centroids' in locals() and len(centroids) > 1:
             ct_arr = np.array(centroids)
             prox_thresh = config.get("PROXIMITY_THRESHOLD", 80)
             for i in range(len(centroids)):
                # Draw person node
                color = (0, 255, 0) # Green default
                if i in clustered_people: color = (0, 0, 255) # Red if clustered
                
                cv2.circle(frame, centroids[i], 5, color, -1)
                
                for j in range(i + 1, len(centroids)):
                    dist = np.linalg.norm(ct_arr[i] - ct_arr[j])
                    if dist < prox_thresh:
                        # Draw line varying from Yellow to Red based on closeness
                        intensity = int(255 * (1 - (dist / prox_thresh)))
                        line_color = (0, intensity, 255) # BGR
                        cv2.line(frame, centroids[i], centroids[j], line_color, 2)

        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        cv2.putText(frame, f'{last_situation}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
        cv2.putText(frame, f'RiskIdx: {int(last_risk_score*100)}% | Cap: {last_person_count}/{config["MAX_CAPACITY"]}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if camera_id not in locks: locks[camera_id] = threading.Lock()
        with locks[camera_id]:
            output_frames[camera_id] = frame.copy()
            status_data[camera_id] = {
                "location": camera_id.upper(), "situation": last_situation, "person_count": last_person_count,
                "avg_motion": round(float(last_avg_motion), 2), "risk_score": round(float(last_risk_score), 2)
            }
        # Limit to ~10 FPS to save CPU
        time.sleep(0.1)

# --- Frame Generator (Full Original Code Restored) ---
def generate_frames(camera_id):
    print(f"DEBUG: Streaming request started for {camera_id}")
    while True:
        if camera_id not in locks: locks[camera_id] = threading.Lock()
        with locks[camera_id]:
            frame = output_frames.get(camera_id)
            if frame is None:
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Initializing...', (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = placeholder
        
        # Encode frame to JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.1) # 10 FPS streaming is sufficient


# ==========================================================
# === NEW & UPDATED FLASK ROUTES WITH LOGIN LOGIC ===
# ==========================================================

@app.route('/')
def welcome():
    # The main page is now the login page
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    # Check if user exists and password is correct
    if username in AUTHORITY_USERS and AUTHORITY_USERS[username] == password:
        session['logged_in'] = True
        session['username'] = username
        # Initialize user data if not exists (for dynamically registered users)
        if username not in USER_DATA:
            USER_DATA[username] = { "sources": {}, "labels": {} }
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html', error='Invalid credentials. Please try again.')

@app.route('/dashboard')
def dashboard():
    # Protect this route
    if not session.get('logged_in'):
        return redirect(url_for('welcome'))
    
    username = session['username']
    user_cameras = USER_DATA.get(username, {}).get("sources", {})
    
    # Pass generic venue info to template
    return render_template('index.html', camera_ids=list(user_cameras.keys()), venue_name=VENUE_INFO.get('name', 'Venue Monitor'), timestamp=int(time.time()))

@app.route('/api/analytics')
def analytics_api():
    if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
    
    # Return predictions for all cameras
    analytics = {}
    for cam_id, config in CAMERA_CONFIGS.items():
        if cam_id in PREDICTOR.history:
             pred = PREDICTOR.get_prediction(cam_id, config["MAX_CAPACITY"])
             analytics[cam_id] = pred
             
    return jsonify(analytics)


@app.route('/admin')
def admin_panel():
    print(f"DEBUG: Accessing /admin. Session: {session}")
    if not session.get('logged_in'): 
        print("DEBUG: User not logged in. Redirecting to welcome.")
        return redirect(url_for('welcome'))
    username = session['username']
    user_data = USER_DATA.get(username, {})
    return render_template('admin.html', 
                         cameras=user_data.get("sources", {}), 
                         labels=user_data.get("labels", {}))

@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
    username = session['username']
    cam_id = request.json.get('id')
    
    # 1. Stop thread
    if cam_id in STOP_EVENTS:
        STOP_EVENTS[cam_id].set()
    
    # 2. Remove from User Data
    if username in USER_DATA:
        USER_DATA[username]["sources"].pop(cam_id, None)
        USER_DATA[username]["labels"].pop(cam_id, None)
        
    return jsonify({"status": "ok"})





@app.route('/add_camera', methods=['POST'])
def add_camera():
    if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
    
    username = session['username']
    data = request.json
    
    # Generate ID
    import uuid
    new_id = f"cam_{str(uuid.uuid4())[:6]}"
    
    # Parse source
    source_type = data.get('type') # 'file', 'webcam', 'rtsp', 'youtube'
    source_val = data.get('source')
    
    final_source = source_val
    if source_type == 'file':
        # Auto-resolve file path
        potential_path = VIDEO_DIR / source_val
        if potential_path.exists():
            final_source = str(potential_path)
    elif source_type == 'webcam':
         if str(source_val).isdigit(): final_source = int(source_val)
    elif source_type == 'youtube':
        try:
            ydl_opts = {'format': 'best'}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_val, download=False)
                final_source = info['url']
        except Exception as e:
            return jsonify({"status": "error", "message": f"YouTube Error: {str(e)}"}), 400

    # Save to user config
    USER_DATA[username]["sources"][new_id] = final_source
    USER_DATA[username]["labels"][new_id] = data.get('label', new_id)
    
    # Update CAMERA_CONFIGS with default area logic
    CAMERA_CONFIGS[new_id] = DEFAULT_CAMERA_CONFIG.copy()
    # If the user provided an area (not currently in UI but good for API), use it, else default
    CAMERA_CONFIGS[new_id]["AREA_SQ_METERS"] = float(data.get('area', 20.0))
    
    # Initialize lock & Stop Event
    locks[new_id] = threading.Lock()
    STOP_EVENTS[new_id] = threading.Event()
    
    # Start thread
    thread = threading.Thread(target=run_stampede_detection, args=(new_id, final_source, shared_model), daemon=True)
    thread.start()
    
    return jsonify({"status": "ok", "id": new_id})

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('welcome'))

@app.route('/pilgrim')
def pilgrim_page():
    # This page is public, no login required
    return render_template('pilgrim.html')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    # Protect this route
    if not session.get('logged_in'):
        return "Access Denied", 401
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status/all')
def status_all():
    # Protect this route
    if not session.get('logged_in'):
        return jsonify({"error": "Unauthorized"}), 401
    
    # Filter status to only show what the user owns
    # (Optional security step, but good for multi-user)
    username = session['username']
    user_sources = USER_DATA.get(username, {}).get("sources", {})
    
    filtered_status = { k:v for k,v in status_data.items() if k in user_sources }
    return jsonify(filtered_status)

# --- Add this new route to your app.py ---


@app.route('/api/public_status')
def public_status():
    """
    A public endpoint to provide status data for the pilgrim page.
    This returns the full status dictionary.
    """
    # Using a lock ensures we don't read the data while it's being written
    # Fallback if no keys exist to avoid index error
    if not locks:
        return jsonify(status_data)
        
    first_key = list(locks.keys())[0]
    with locks[first_key]:
        return jsonify(status_data)


@app.route('/api/gate_status')
def gate_status():
    # This API is public for the pilgrim page
    gate_statuses = {}
    # Use ALL_VIDEO_SOURCES or locks
    if not locks:
         return jsonify(gate_statuses)

    first_key = list(locks.keys())[0]
    with locks[first_key]:
        for gate, cam_id in GATE_CAMERA_MAPPING.items():
            situation = status_data.get(cam_id, {}).get("situation", "Initializing...")
            gate_statuses[gate] = situation
    return jsonify(gate_statuses)


@app.route('/register', methods=['POST'])
def register():
    username = request.form['new_username']
    password = request.form['new_password']
    email = request.form['email']

        # TEMP: store in session (for hackathon demo)
    session['logged_in'] = True
    session['username'] = username

    print("New User Registered:", username, email)

    # After registering → show Add Camera form in the same page
    return render_template("login.html", show_add_camera=True)
# --- Main (Full Original Code Restored) ---
if __name__ == '__main__':
    print("Loading shared YOLO model...")
    shared_model = YOLO("yolov8n.onnx", task="detect")

    print("Model loaded.")

    print("Model loaded.")

    for cam_id, path in ALL_VIDEO_SOURCES.items():
        if not os.path.exists(str(path)) and not str(path).isdigit() and not str(path).startswith("http"):
             # Simple check to warn only if it looks like a file path that is missing
             if "mp4" in str(path) or "avi" in str(path):
                 print(f"WARNING: Video not found for {cam_id} at {path}")
                 continue
        
        STOP_EVENTS[cam_id] = threading.Event()
        thread = threading.Thread(target=run_stampede_detection, args=(cam_id, path, shared_model), daemon=True)
        thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
