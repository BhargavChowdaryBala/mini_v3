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

def deskew_plate(image):
    """Corrects the skew/tilt of a license plate crop using minAreaRect."""
    if image is None or image.size == 0:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binary threshold and finding contours to estimate rotation
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    
    if len(coords) < 10: return image # Not enough data
    
    angle = cv2.minAreaRect(coords)[-1]
    # Adjust angle for minAreaRect output format
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def apply_morphology(image):
    """Bridge character gaps using morphological closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_professional_restoration(image):
    """
    Elite Image Signal Processing (ISP) pipeline for ANPR.
    Returns 3 versions for Multi-Pass OCR.
    """
    if image is None or image.size == 0:
        return [image]
    
    # 1. Bicubic Super-Resolution (Upscale to target 150px height)
    h, w = image.shape[:2]
    scale = 200 / h if h < 200 else 1.0
    upscaled = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    
    # 2. NLMeans Denoising (Preserves edges while removing digital artifacts)
    denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
    
    # 3. CLAHE (Local Contrast Enhancement)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 4. Unsharp Masking (Sharpening)
    gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
    
    # Conversion to Grayscale for final passes
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    
    # Pass A: Normalized Grayscale
    pass_a = gray.copy()
    
    # Pass B: Otsu Binarization + Morphology
    _, pass_b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Auto-Invert if needed
    if cv2.countNonZero(pass_b) < (pass_b.shape[0] * pass_b.shape[1] * 0.5):
        pass_b = cv2.bitwise_not(pass_b)
    pass_b = apply_morphology(pass_b)
    
    # Pass C: Adaptive Thresholding
    pass_c = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    if cv2.countNonZero(pass_c) < (pass_c.shape[0] * pass_c.shape[1] * 0.5):
        pass_c = cv2.bitwise_not(pass_c)

    return [pass_a, pass_b, pass_c]

def character_voting(plate_results):
    """
    Performs character-level voting across multiple plate strings with confidence weighting.
    plate_results: List of (text, confidence) tuples
    """
    if not plate_results:
        return "UNKNOWN"
    
    # Filter out UNKNOWN or very short results
    valid_results = [(s.upper(), conf) for s, conf in plate_results if s != "UNKNOWN" and len(s) >= 4]
    if not valid_results:
        return "UNKNOWN"
    
    # Find the most frequent length
    lengths = [len(s) for s, _ in valid_results]
    target_len = Counter(lengths).most_common(1)[0][0]
    
    # Filter strings to target length (or close to it)
    voters = [(s, conf) for s, conf in valid_results if abs(len(s) - target_len) <= 1]
    
    final_plate = ""
    for i in range(target_len):
        char_weights = {} # {char: total_confidence}
        for s, conf in voters:
            if i < len(s):
                char = s[i]
                char_weights[char] = char_weights.get(char, 0) + conf
        
        if char_weights:
            # Pick character with highest combined confidence
            final_char = max(char_weights, key=char_weights.get)
            final_plate += final_char
            
    return final_plate

def extract_indian_number_plate(text_list):
    """
    Standardized Indian Number Plate extraction logic.
    Prioritizes 10-character Indian format: [LL][NN][LL][NNNN]
    - LL: State Code (e.g. AP)
    - NN: District Code (e.g. 39)
    - LL: Serial Code (e.g. UX)
    - NNNN: Unique Number (e.g. 8273)
    """
    text = " ".join(text_list).upper()
    # Remove common OCR noise
    text = re.sub(r'\bIND\b|\bND\b|\s+', '', text)
    # Remove all non-alphanumeric chars
    text = re.sub(r'[^A-Z0-9]', '', text)
    # Safe OCR corrections (most common mistakes)
    # Be careful not to replace every O/0 if it's at the start of a serial etc,
    # but for most Indian plates, this helps significantly.
    text = text.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')

    # [STRICT 10-CHAR] ✅ Standard Modern (e.g., AP39UX8273)
    # Pattern: [2 Letters][2 Numbers][2 Letters][4 Numbers]
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', text)
    if match: return match.group()

    # [STRICT 10-CHAR-B] ✅ (e.g., AP39 0442 -> maybe missed serial, but if 10 is the rule)
    # If the user says it has 10, but detection is flaky:
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}', text)
    if match: 
        candidate = match.group()
        if len(candidate) == 10: return candidate

    # ✅ Bharat Series (e.g., 22BH1234AA) - 10 chars
    match = re.search(r'\d{2}BH\d{4}[A-Z]{2}', text)
    if match: return match.group()

    # ✅ Standard Modern 9-character fallback (e.g. KA01M1234)
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{1}\d{4}', text)
    if match: return match.group()

    # ✅ Older Format / 2-digit serial / Govt (e.g., DL3CA6341)
    match = re.search(r'[A-Z]{2}\d{1,2}[A-Z]?\d{4}', text)
    if match: return match.group()
    
    # ✅ Very General Alphanumeric fallback (strictly 10 chars if possible)
    # If we have a 10 char string that looks like a plate
    match = re.search(r'[A-Z0-9]{10}', text)
    if match: return match.group()

    # Final Catch-all (5-12 chars)
    match = re.search(r'[A-Z0-9]{5,12}', text)
    if match:
        candidate = match.group()
        # Custom logic for "COLLEGEBUS" and similar detections
        if len(candidate) > 7 and candidate.isalpha():
            return candidate
        if sum(c.isdigit() for c in candidate) >= 2 and sum(c.isalpha() for c in candidate) >= 2:
            return candidate

    return "UNKNOWN"

class BusProcessor:
    def __init__(self, bus_model_path=None, plate_model_path=None, line_position=0.5, line_direction='vertical'):
        """
        Refactored for Video File Pipeline.
        - line_position: 0.0 to 1.0 (percent of width/height)
        - line_direction: 'vertical' (for left-right) or 'horizontal' (for top-bottom)
        """
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
        # Trigger OCR even when the bus is relatively far (0.8% of frame area)
        self.proximity_threshold_ratio = 0.008 
        self.capture_count = 15 # Increased for a bigger voting pool
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

            for box, track_id_raw, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id_raw) # Ensure it's a python int
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                # Draw bus detection with persistent ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Bus {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 1. Crossing Detection
                if track_id not in self.processed_ids:
                    # Initialize state for new IDs
                    if track_id not in self.proximity_states:
                        self.proximity_states[track_id] = {"crossed": False, "frames": [], "analyzed": False}
                    
                    state = self.proximity_states[track_id]
                    
                    # Detect Crossing
                    if track_id in self.tracking_history:
                        prev_pos = self.tracking_history[track_id]
                        line_val = (w * self.line_position) if self.line_direction == 'vertical' else (h * self.line_position)
                        curr_pos = centroid_x if self.line_direction == 'vertical' else centroid_y
                        
                        if (prev_pos < line_val <= curr_pos) or (prev_pos > line_val >= curr_pos):
                            state["crossed"] = True
                            print(f"[AI] Bus {track_id} crossed virtual line.")

                    # 2. Trigger Burst Capture (Once crossed AND near enough OR if already very large)
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / (w * h)

                    # Trigger if crossed OR if bus is already very large (fallback for videos starting mid-way)
                    should_trigger = state["crossed"] or area_ratio > 0.15

                    if should_trigger and not state["analyzed"]:
                        if area_ratio > self.proximity_threshold_ratio or len(state["frames"]) > 0:
                            if len(state["frames"]) == 0:
                                print(f"[AI] >>> START CAPTURE for Bus {track_id} (Ratio: {area_ratio:.3f})")
                            
                            # Start or continue capture
                            if len(state["frames"]) < self.capture_count:
                                state["frames"].append(frame.copy())
                                cv2.circle(annotated_frame, (x1+15, y1+15), 8, (0, 0, 255), -1)
                            
                            if len(state["frames"]) == self.capture_count:
                                state["analyzed"] = True
                                self.processed_ids.add(track_id)
                                
                                print(f"[AI] >>> ANALYSIS TRIGGERED for Bus {track_id}")
                                thread = threading.Thread(target=self._background_burst_analysis, 
                                                         args=(state["frames"], "DETECTED", track_id))
                                thread.daemon = True
                                thread.start()

                    # Update history for next frame
                    self.tracking_history[track_id] = centroid_x if self.line_direction == 'vertical' else centroid_y

        return annotated_frame

    def _background_burst_analysis(self, burst_frames, direction, track_id):
        """
        Processes captured frames with intelligence:
        1. Localize plate in ALL frames.
        2. Sort by sharpness and pick the top 8 clearest crops.
        3. Perform deskewing and bilateral filtering.
        4. Run OCR and vote.
        """
        plate_candidates = []
        
        for frame in burst_frames:
            with self.lock:
                plate_results = self.plate_model.predict(frame, conf=0.3, verbose=False)
            
            if len(plate_results[0].boxes) > 0:
                box = plate_results[0].boxes.xyxy[0].cpu().numpy()
                px1, py1, px2, py2 = map(int, box)
                plate_crop = frame[py1:py2, px1:px2]
                
                if plate_crop.size > 0:
                    sharpness = calculate_sharpness(plate_crop)
                    plate_candidates.append({"img": plate_crop, "sharpness": sharpness})

        if plate_candidates:
            # Sort by sharpness (Descending) and take top 8
            plate_candidates.sort(key=lambda x: x["sharpness"], reverse=True)
            top_crops = [x["img"] for x in plate_candidates[:8]]
            
            # Additional Preprocessing: Deskewing
            deskewed_crops = [deskew_plate(img) for img in top_crops]
            
            self.run_multi_ocr(deskewed_crops, direction, track_id)
        else:
            print(f"[AI] No plate localized in burst for Bus {track_id}.")
            log_event("numberplate missed", "DETECTED", source=self.current_session, bus_id=track_id)

    def run_multi_ocr(self, plate_images, direction, track_id):
        if self.ocr is None: return

        results = [] # Store (text, confidence)
        # [Smart Multi-Pass] Step 1: Fast Grayscale Pass on all frames
        for img in plate_images:
            # We already have deskewed crops here
            h, w = img.shape[:2]
            scale = 200 / h if h < 200 else 1.0
            prepped = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            prepped = cv2.fastNlMeansDenoisingColored(prepped, None, 10, 10, 7, 21)
            
            gray = cv2.cvtColor(prepped, cv2.COLOR_BGR2GRAY)
            try:
                with self.lock:
                    result = self.ocr.ocr(gray, cls=True)
                if result and result[0]:
                    lines = [line[1] for idx in range(len(result)) for line in result[idx]]
                    raw_text = "".join([l[0] for l in lines])
                    conf = sum([l[1] for l in lines]) / len(lines) if lines else 0
                    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                    if len(clean) >= 4: results.append((clean, conf))
            except: pass

        # Step 2: Fallback to Heavy Passes only if needed
        if not results or max([r[1] for r in results]) < 0.6:
            # Pick the top 5 frames for heavy processing
            for img in plate_images[-5:]:
                versions = apply_professional_restoration(img)
                # apply_professional_restoration returns [pass_a, pass_b, pass_c]
                # Pass B (Otsu) and Pass C (Adaptive) are the "heavy" ones
                for processed_img in versions[1:]: 
                    try:
                        with self.lock:
                            result = self.ocr.ocr(processed_img, cls=True)
                        if result and result[0]:
                            lines = [line[1] for idx in range(len(result)) for line in result[idx]]
                            raw_text = "".join([l[0] for l in lines])
                            conf = sum([l[1] for l in lines]) / len(lines) if lines else 0
                            clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
                            if len(clean) >= 4: results.append((clean, conf))
                    except: pass

        final_plate_raw = character_voting(results)
        
        # Apply extraction regex
        final_plate = extract_indian_number_plate([final_plate_raw])
        
        # [NEW] Fallback: If voting failed, pick the single highest-confidence raw result
        if (final_plate == "UNKNOWN" or len(final_plate) < 4) and results:
            best_single = max(results, key=lambda x: x[1])
            if best_single[1] > 0.6: # If confidence > 60%
                final_plate = extract_indian_number_plate([best_single[0]])
        
        if final_plate != 'UNKNOWN':
            log_event(final_plate, 'DETECTED', source=self.current_session, bus_id=track_id)
        else:
            # If all 5 frames failed to produce a valid OCR result
            log_event('numberplate missed', 'DETECTED', source=self.current_session, bus_id=track_id)

    def run_ocr(self, plate_img, direction):
        """Legacy support for single plate processing."""
        self.run_multi_ocr([plate_img], 'DETECTED')

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
