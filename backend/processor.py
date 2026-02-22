import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from database import log_event
import time

class BusProcessor:
    def __init__(self, bus_model_path='yolov8n.pt', plate_model_path='best.pt', line_y=400):
        # Load models
        print("Loading Bus Model...")
        self.bus_model = YOLO(bus_model_path)
        print("Bus Model Loaded.")
        print("Loading Plate Model...")
        self.plate_model = YOLO(plate_model_path)
        print("Plate Model Loaded.")
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        print("PaddleOCR Initialized.")
        
        # Virtual line position (y-coordinate)
        self.line_y = line_y
        
        # Tracking and event locking
        self.processed_ids = set()
        self.tracking_history = {} # id: last_y
        self.ready_for_ocr = {} # id: countdown (frames to capture)
        self.captured_frames = {} # id: list of frames
        
        # Status
        self.system_status = "System Ready"

    def process_frame(self, frame):
        # Detect and track buses
        # Detect only 'bus' (class 5 in COCO)
        results = self.bus_model.track(frame, persist=True, classes=[5], tracker="bytetrack.yaml", verbose=False)
        
        annotated_frame = results[0].plot()
        
        # Draw virtual lines (reversing camera style)
        h, w, _ = frame.shape
        
        # Layout points
        bottom_y = self.line_y
        near_y = int(self.line_y * 0.85)
        mid_y = int(self.line_y * 0.7)
        far_y = int(self.line_y * 0.5)
        
        # Red line (Gate Line) - Closest
        cv2.line(annotated_frame, (int(w*0.05), bottom_y), (int(w*0.95), bottom_y), (0, 0, 255), 3)
        cv2.putText(annotated_frame, "GATE LINE: RED ZONE", (int(w*0.05), bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Yellow lines (Perspective Guidelines)
        cv2.line(annotated_frame, (int(w*0.15), near_y), (int(w*0.85), near_y), (0, 255, 255), 2)
        cv2.line(annotated_frame, (int(w*0.25), mid_y), (int(w*0.75), mid_y), (0, 255, 255), 2)
        cv2.line(annotated_frame, (int(w*0.35), far_y), (int(w*0.65), far_y), (0, 255, 255), 2)
        
        # Diagonal side lines
        cv2.line(annotated_frame, (int(w*0.05), bottom_y), (int(w*0.35), far_y), (0, 255, 255), 2)
        cv2.line(annotated_frame, (int(w*0.95), bottom_y), (int(w*0.65), far_y), (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                centroid_y = int((y1 + y2) / 2)
                
                # Check for line crossing
                if track_id in self.tracking_history:
                    prev_y = self.tracking_history[track_id]
                    
                    # Direction check
                    direction = None
                    if prev_y < self.line_y <= centroid_y:
                        direction = "EXIT" # Moving down
                    elif prev_y > self.line_y >= centroid_y:
                        direction = "ENTRY" # Moving up
                    
                    if direction and track_id not in self.processed_ids:
                        print(f"Bus {track_id} crossed line: {direction}")
                        self.system_status = f"Processing Bus: {direction}"
                        # Lock event and start OCR capture
                        self.ready_for_ocr[track_id] = {"count": 3, "direction": direction}
                        self.captured_frames[track_id] = []
                        self.processed_ids.add(track_id)
                
                # Update history
                self.tracking_history[track_id] = centroid_y
                
                # Handle OCR frame capture
                if track_id in self.ready_for_ocr:
                    if self.ready_for_ocr[track_id]["count"] > 0:
                        # Capture frame for OCR
                        self.captured_frames[track_id].append(frame.copy())
                        self.ready_for_ocr[track_id]["count"] -= 1
                    else:
                        # Trigger OCR on the best frame (simplified: middle frame)
                        best_frame = self.captured_frames[track_id][1]
                        self.process_ocr(best_frame, self.ready_for_ocr[track_id]["direction"])
                        del self.ready_for_ocr[track_id]
                        del self.captured_frames[track_id]

        return annotated_frame

    def process_ocr(self, frame, direction):
        # Detect plate
        plate_results = self.plate_model(frame, verbose=False)
        if len(plate_results[0].boxes) > 0:
            # Get best plate box
            box = plate_results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Crop plate
            plate_img = frame[y1:y2, x1:x2]
            
            # OCR
            ocr_result = self.ocr.ocr(plate_img, cls=True)
            if ocr_result and ocr_result[0]:
                plate_text = ""
                for line in ocr_result[0]:
                    plate_text += line[1][0]
                
                # Clean text (remove spaces, special chars)
                plate_text = "".join(e for e in plate_text if e.isalnum()).upper()
                
                if plate_text:
                    log_event(plate_text, direction)
                    self.system_status = "System Ready"
        else:
            print("No plate detected in captured frames.")
            self.system_status = "System Ready"

    def get_status(self):
        return self.system_status
