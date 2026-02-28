import os
# AGGRESSIVELY disable oneDNN/MKLDNN to prevent "OneDnnContext does not have the input Filter" crash on Windows
os.environ['PADDLE_ONEDNN_DISABLE'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_mkldnn_bfloat16'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from database import log_event
import time
import threading
import re
import gc
from collections import Counter

def calculate_sharpness(image):
    """Calculates the sharpness of an image using Laplacian variance."""
    if image is None or image.size == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def apply_clahe(image):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def character_voting(plate_strings):
    """
    Performs character-level voting across multiple plate strings.
    Assumes strings are somewhat aligned or of similar length.
    """
    if not plate_strings:
        return "UNKNOWN"
    
    # Filter out UNKNOWN or very short results
    valid_strings = [s for s in plate_strings if s != "UNKNOWN" and len(s) >= 4]
    if not valid_strings:
        return "UNKNOWN"
    
    # Find the most frequent length
    lengths = [len(s) for s in valid_strings]
    target_len = Counter(lengths).most_common(1)[0][0]
    
    # Filter strings to target length (or close to it)
    voters = [s for s in valid_strings if abs(len(s) - target_len) <= 1]
    
    final_plate = ""
    for i in range(target_len):
        chars_at_pos = []
        for s in voters:
            if i < len(s):
                chars_at_pos.append(s[i])
        
        if chars_at_pos:
            common = Counter(chars_at_pos).most_common(1)
            if common:
                final_plate += common[0][0]
            
    return final_plate

def extract_indian_number_plate(text_list):
    """Standardized Indian Number Plate extraction logic."""
    text = " ".join(text_list).upper()
    # Remove common OCR noise
    text = re.sub(r'\bIND\b|\bND\b', '', text)
    # Remove all non-alphanumeric chars
    text = re.sub(r'[^A-Z0-9]', '', text)
    # Safe OCR corrections
    text = text.replace('O', '0').replace('I', '1')

    # ✅ Bharat Series: YYBHXXXX + optional letters
    match = re.search(r'\d{2}BH\d{4}[A-Z]{0,2}', text)
    if match: return match.group()

    # ✅ Normal Indian plates fallback
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{0,2}\d{4}', text)
    if match: return match.group()

    # ✅ Generic fallback for partial/custom plates
    match = re.search(r'[A-Z0-9]{6,12}', text)
    if match: return match.group()

    return "UNKNOWN"

class BusProcessor:
    def __init__(self, bus_model_path=None, plate_model_path=None, line_position=0.5, line_direction='vertical'):
        """
        Refactored for Video File Pipeline.
        - line_position: 0.0 to 1.0 (percent of width/height)
        - line_direction: 'vertical' (for left-right) or 'horizontal' (for top-bottom)
        """
        # Resolve paths relative to this file
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        if bus_model_path is None:
            bus_model_path = os.path.join(base_path, 'yolov8n.pt')
        if plate_model_path is None:
            plate_model_path = os.path.join(base_path, 'best.pt')
            
        # Virtual line configuration
        self.line_position = line_position
        self.line_direction = line_direction
        
        print(f"Loading Models (Bus: {bus_model_path}, Plate: {plate_model_path})...")
        self.bus_model = YOLO(bus_model_path)
        self.bus_model.to('cpu')
        self.plate_model = YOLO(plate_model_path)
        self.plate_model.to('cpu')
        gc.collect()
        
        print("Initializing PaddleOCR...")
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                ir_optim=False, 
                enable_mkldnn=False,
                use_tensorrt=False,
                show_log=False
            )
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.ocr = None
        
        self.system_status = "System Ready"
        
        # State tracking for File Pipeline
        self.lock = threading.Lock() 
        
        # State tracking for File Pipeline
        self.tracking_history = {} # {id: last_pos}
        self.processed_ids = set() # {id}
        self.proximity_states = {} # {id: {"crossed": bool, "frames": [], "best_size": 0}}
        
        # Proximity settings
        # Area ratio to trigger OCR (bus takes up 8% of frame is usually very close)
        self.proximity_threshold_ratio = 0.08 
        self.capture_count = 5 # Exactly 5 frames as requested
        self.current_session = "live"

    def set_session(self, name):
        """Sets a new storage session name (for logging context only)."""
        # Sanitize name
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        self.current_session = clean_name
        return clean_name
        

    def extract_indian_number_plate(self, text_list):
        return extract_indian_number_plate(text_list)

    def process_frame(self, frame):
        """
        Refactored Frame Processing:
        1. Tracking with ByteTrack (via YOLO model)
        2. Crossing Detection (Virtual Line)
        3. Trigger Capture (Exactly 5 frames when front part is near camera)
        4. Trigger Analysis
        """
        with self.lock:
            # Detect and Track Buses (class 5)
            # persistent=True ensures ByteTrack is active
            results = self.bus_model.track(frame, persist=True, classes=[5], conf=0.35, tracker="bytetrack.yaml", verbose=False)
        
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Draw Virtual Gate Line (User's virtual line enabled on screen)
        if self.line_direction == 'vertical':
            line_x = int(w * self.line_position)
            cv2.line(annotated_frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "VIRTUAL LINE", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            line_y = int(h * self.line_position)
            cv2.line(annotated_frame, (0, line_y), (w, line_y), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "VIRTUAL LINE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                # Draw bus detection with persistent ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Bus {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 1. Crossing & Nearness Detection
                if track_id not in self.processed_ids:
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / (w * h)
                    
                    # Logic: Capture when bus is crossing the line AND large enough (front is near camera)
                    is_crossing = False
                    if track_id in self.tracking_history:
                        prev_pos = self.tracking_history[track_id]
                        line_val = (w * self.line_position) if self.line_direction == 'vertical' else (h * self.line_position)
                        curr_pos = centroid_x if self.line_direction == 'vertical' else centroid_y
                        
                        if (prev_pos < line_val <= curr_pos) or (prev_pos > line_val >= curr_pos):
                            is_crossing = True
                    
                    if is_crossing and area_ratio > self.proximity_threshold_ratio:
                        if track_id not in self.proximity_states:
                            self.proximity_states[track_id] = {"frames": [], "analyzed": False}
                    
                    # 2. Capture exactly 5 frames
                    if track_id in self.proximity_states and not self.proximity_states[track_id]["analyzed"]:
                        state = self.proximity_states[track_id]
                        if len(state["frames"]) < self.capture_count:
                            state["frames"].append(frame.copy())
                        
                        if len(state["frames"]) == self.capture_count:
                            state["analyzed"] = True
                            self.processed_ids.add(track_id)
                            
                            print(f"[AI] Capturing 5 frames for Bus {track_id}. Triggering analysis.")
                            thread = threading.Thread(target=self._background_burst_analysis, 
                                                     args=(state["frames"], "UNKNOWN", track_id))
                            thread.daemon = True
                            thread.start()

                    # Update history for next frame
                    self.tracking_history[track_id] = centroid_x if self.line_direction == 'vertical' else centroid_y

        return annotated_frame

    def _background_burst_analysis(self, burst_frames, direction, track_id):
        """
        Processes exactly 5 captured frames:
        1. Detect plate in each frame using best.pt (plate_model)
        2. Run PaddleOCR on each detected plate
        3. Store result in MongoDB via character voting
        """
        plate_crops = []
        
        for frame in burst_frames:
            with self.lock:
                # Detect Plate using best.pt
                plate_results = self.plate_model.predict(frame, conf=0.3, verbose=False)
            
            if len(plate_results[0].boxes) > 0:
                box = plate_results[0].boxes.xyxy[0].cpu().numpy()
                px1, py1, px2, py2 = map(int, box)
                plate_crop = frame[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    plate_crops.append(plate_crop)

        if plate_crops:
            self.run_multi_ocr(plate_crops, direction, track_id)
        else:
            print(f"[AI] No plate detected in any of the 5 frames for Bus {track_id}.")
            log_event("numberplate missed", direction, source=self.current_session, bus_id=track_id)

    def run_multi_ocr(self, plate_images, direction, track_id):
        if self.ocr is None: return

        results = []
        for img in plate_images:
            processed_img = apply_clahe(img)
            try:
                with self.lock:
                    result = self.ocr.ocr(processed_img, cls=True)
                if result and result[0]:
                    raw_texts = [line[1][0] for idx in range(len(result)) for line in result[idx]]
                    plate_text = extract_indian_number_plate(raw_texts)
                    if plate_text != "UNKNOWN":
                        results.append(plate_text)
            except Exception as e:
                print(f"OCR Error: {e}")
        
        # Character-level voting across valid results
        final_plate = character_voting(results)
        
        if final_plate != "UNKNOWN":
            log_event(final_plate, direction, source=self.current_session, bus_id=track_id)
        else:
            # If all 5 frames failed to produce a valid OCR result
            log_event("numberplate missed", direction, source=self.current_session, bus_id=track_id)

    def run_ocr(self, plate_img, direction):
        """Legacy support for single plate processing."""
        self.run_multi_ocr([plate_img], direction)

    def get_status(self):
        return self.system_status

    def reset(self):
        self.system_status = "System Ready"
        self.tracking_history = {}
        self.processed_ids = set()
        self.proximity_states = {}
        print("[Processor] State Reset for new video loop.")

class VideoUploadProcessor:
    def __init__(self, bus_model_path=None, plate_model_path=None, line_position=0.5, line_direction='horizontal'):
        """
        Unified Video Upload Processor.
        Uses a BusProcessor instance to maintain consistency in logic.
        """
        self.processor = BusProcessor(
            bus_model_path=bus_model_path,
            plate_model_path=plate_model_path,
            line_position=line_position,
            line_direction=line_direction
        )
        self.lock = threading.Lock()

    def process_video(self, video_path, progress_callback=None, frame_callback=None):
        """Processes an uploaded video file using the unified proximity logic."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Error: Could not determine frame count for {video_path}")
            cap.release()
            return
            
        frame_idx = 0
        
        # Set a unique session name for this upload to organize frames in subfolders
        video_name = os.path.basename(video_path).split('.')[0]
        timestamp = int(time.time())
        self.processor.set_session(f"upload_{video_name}_{timestamp}")
        
        # Reset processor state for this specific video run
        self.processor.tracking_history = {}
        self.processor.processed_ids = set()
        self.processor.proximity_states = {}

        print(f"[VideoMode] Starting analysis of {video_path} (Unified Proximity Logic)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_idx += 1
            
            # Use the core logic from BusProcessor
            # This handles detection, tracking, crossing, and proximity-based OCR triggering
            annotated_frame = self.processor.process_frame(frame)

            if frame_callback:
                frame_callback(annotated_frame)

            if frame_idx % 20 == 0 and progress_callback:
                progress_callback(int((frame_idx / total_frames) * 100))

        cap.release()
        print(f"[VideoMode] Finished analysis of {video_path}")

    def _extract_plate(self, text_list):
        return extract_indian_number_plate(text_list)
