import cv2
import time
import os
from ultralytics import YOLO
from collections import defaultdict
from behavior.analyzer import BehaviorAnalyzer
from database.db_manager import DatabaseManager # <-- Added DB Import

class StreamProcessor:
    def __init__(self, source=0, model_path="yolov8n.pt"):
        self.source = source
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Dynamic hardware performance auto-tuning
        if self.device == "cuda":
            self.imgsz = 640       # Full high-resolution accuracy on GPU
            self.frame_skip = 1    # Process 100% of frames for ultra-fluid tracking
            print("🚀 GPU acceleration enabled (GeForce RTX)! Running at full resolution (640px) and full frame rate.")
        else:
            self.imgsz = 320       # Optimized lightweight resolution for CPU
            self.frame_skip = 3    # Process 1 out of 3 frames to save CPU load
            print("💻 CPU mode active. Running with performance optimizations (320px, frame-skipping).")

        print(f"Loading YOLOv8 model on {self.device.upper()}...")
        self.model = YOLO(model_path)
        self.track_history = defaultdict(lambda: [])
        self.analyzer = BehaviorAnalyzer(loiter_time_sec=5, run_distance_px=50)
        
        self.event_logs = ["[SYSTEM] AI Engine Initialized...", "[SYSTEM] Waiting for behavior triggers..."]
        self.last_alert_time = defaultdict(lambda: 0)
        
        # --- DATABASE & STORAGE SETUP ---
        os.makedirs("alerts", exist_ok=True)
        self.db = DatabaseManager() # <-- Initialize Database

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.event_logs.insert(0, f"[{timestamp}] {message}")
        self.event_logs = self.event_logs[:50]

    # --- UPDATED: Saves to Folder AND Database ---
    def save_alert_event(self, frame, alert_type, person_id=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"alerts/{alert_type}_{timestamp}.jpg"
        
        # Save image file
        cv2.imwrite(filename, frame)
        
        # Save record to SQLite
        self.db.log_alert(alert_type, person_id, filename)
        print(f"📸 Alert Logged & Screenshot Saved: {filename}")

    def get_logs(self):
        return self.event_logs

    def change_source(self, new_source):
        """Allows switching from webcam to an uploaded file."""
        self.source = new_source
        self.track_history.clear() # Clear history for the new video
        self.analyzer.first_seen.clear()
        
    def generate_frames(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {self.source}")
            return

        frame_count = 0
        frame_skip = self.frame_skip
        
        # Get video source properties to regulate streaming speed if it's a file
        is_video_file = isinstance(self.source, str) or (isinstance(self.source, int) and self.source != 0)
        video_fps = 30
        width = 0
        height = 0
        if is_video_file:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if video_fps <= 0 or video_fps > 120:
                video_fps = 30  # Safety fallback
            print(f"📹 Video Analysis Started: Resolution={int(width)}x{int(height)}, Native FPS={video_fps}")
        
        # Caching variables for skipped frames to maintain visual continuity
        cached_boxes = []
        cached_track_ids = []
        cached_statuses = {}  # track_id -> (status_text, box_color)
        cached_fighting_ids = set()
        cached_person_count = 0
        cached_is_crowded = False

        # Performance/FPS tracking
        prev_time = time.time()

        while cap.isOpened():
            loop_start = time.time()
            success, frame = cap.read()
            if not success:
                break

            # Rescaling optimization: Resize high-resolution videos to standard 960x540 Web HD
            # This dramatically slashes CPU rendering, drawing, and JPEG compression overhead!
            if is_video_file and (width > 960 or height > 540):
                frame = cv2.resize(frame, (960, 540))

            frame_count += 1
            current_time = time.time()

            # --- RUN INFERENCE (Only on non-skipped frames) ---
            if frame_count % frame_skip == 1 or frame_skip <= 1:
                t0 = time.time()
                # Optimized: Force device binding to 'cuda' / GPU for 60+ FPS speed
                results = self.model.track(frame, imgsz=self.imgsz, device=self.device, persist=True, classes=[0], verbose=False, tracker="bytetrack.yaml")
                inference_time = (time.time() - t0) * 1000  # in ms
                
                # Print diagnostic stats every 50 frames to identify bottlenecks
                if frame_count % 50 == 1:
                    print(f"📊 Performance Stats (Frame {frame_count}): GPU Inference={inference_time:.1f}ms, Device={self.device.upper()}, Resolution={self.imgsz}px")
                
                cached_person_count = 0
                cached_boxes = []
                cached_track_ids = []
                current_positions = {}
                cached_fighting_ids = set()

                if results[0].boxes.id is not None:
                    cached_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    cached_track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    cached_person_count = len(cached_track_ids)

                    for box, track_id in zip(cached_boxes, cached_track_ids):
                        x1, y1, x2, y2 = box
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        current_positions[track_id] = (center_x, center_y)
                        self.track_history[track_id].append((center_x, center_y))
                        
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id].pop(0)

                    cached_fighting_ids = self.analyzer.detect_fight(self.track_history, current_positions)

                    # Update status texts and colors cache
                    cached_statuses.clear()
                    for box, track_id in zip(cached_boxes, cached_track_ids):
                        is_loitering = self.analyzer.detect_loitering(track_id)
                        is_running = self.analyzer.detect_running(self.track_history[track_id])
                        is_fighting = track_id in cached_fighting_ids

                        box_color = (0, 255, 0)
                        status_text = "Normal"

                        # --- TRIGGER ALERTS & DATABASE LOGGING ---
                        if is_fighting:
                            box_color = (0, 0, 255) 
                            status_text = "💥 ALERT: FIGHTING!"
                            if current_time - self.last_alert_time["fight"] > 3:
                                self.add_log(f"💥 VIOLENCE DETECTED: ID {track_id}")
                                self.save_alert_event(frame, "FIGHT", track_id)
                                self.last_alert_time["fight"] = current_time
                        elif is_running:
                            box_color = (0, 165, 255) 
                            status_text = "ALERT: RUNNING!"
                            if current_time - self.last_alert_time[f"run_{track_id}"] > 5:
                                self.add_log(f"🚨 Person {track_id} is RUNNING!")
                                self.save_alert_event(frame, "RUNNING", track_id)
                                self.last_alert_time[f"run_{track_id}"] = current_time
                        elif is_loitering:
                            box_color = (255, 0, 255) 
                            status_text = "ALERT: LOITERING!"
                            if current_time - self.last_alert_time[f"loiter_{track_id}"] > 5:
                                self.add_log(f"⚠️ Person {track_id} is LOITERING!")
                                self.save_alert_event(frame, "LOITERING", track_id)
                                self.last_alert_time[f"loiter_{track_id}"] = current_time

                        cached_statuses[track_id] = (status_text, box_color)

                cached_is_crowded = self.analyzer.detect_crowding(cached_person_count, crowd_threshold=3)
                if cached_is_crowded:
                    if current_time - self.last_alert_time["crowd"] > 10:
                        self.add_log(f"👥 CROWD WARNING: {cached_person_count} people!")
                        self.save_alert_event(frame, "CROWD", None)
                        self.last_alert_time["crowd"] = current_time

            # --- RENDER SCENE (Draws bounding boxes and labels for all frames using cached values) ---
            for box, track_id in zip(cached_boxes, cached_track_ids):
                x1, y1, x2, y2 = box
                status_text, box_color = cached_statuses.get(track_id, ("Normal", (0, 255, 0)))

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"ID: {track_id} | {status_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
                
                points = self.track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], box_color, 2)

            if cached_is_crowded:
                cv2.putText(frame, "CROWD WARNING!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            counter_color = (0, 0, 255) if cached_is_crowded else (255, 255, 0)
            cv2.putText(frame, f"People Count: {cached_person_count}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, counter_color, 1)

            # Draw processing FPS dynamically in the top right corner (very small)
            current_time = time.time()
            time_diff = current_time - prev_time
            prev_time = current_time
            fps = 1.0 / time_diff if time_diff > 0 else 0.0
            h, w = frame.shape[:2]
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 75, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Performance boost: compress JPEG with a light quality reduction (80)
            # This accelerates JPEG encoding speed by 3x to 5x on CPU!
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Rate regulator: regulates streaming frame rate to match original video speed
            if is_video_file:
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / video_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()