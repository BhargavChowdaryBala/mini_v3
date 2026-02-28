import cv2
import os

def capture_and_overlay(video_path, line_y, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    h, w, _ = frame.shape
    
    # Layout points matching processor.py
    bottom_y = line_y
    near_y = int(line_y * 0.85)
    mid_y = int(line_y * 0.7)
    far_y = int(line_y * 0.5)
    
    # Red line (Gate Line)
    cv2.line(frame, (int(w*0.05), bottom_y), (int(w*0.95), bottom_y), (0, 0, 255), 4)
    cv2.putText(frame, f"GATE LINE: {bottom_y}", (int(w*0.05), bottom_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Yellow lines
    cv2.line(frame, (int(w*0.15), near_y), (int(w*0.85), near_y), (0, 255, 255), 2)
    cv2.line(frame, (int(w*0.25), mid_y), (int(w*0.75), mid_y), (0, 255, 255), 2)
    cv2.line(frame, (int(w*0.35), far_y), (int(w*0.65), far_y), (0, 255, 255), 2)
    
    # Diagonal side lines
    cv2.line(frame, (int(w*0.05), bottom_y), (int(w*0.35), far_y), (0, 255, 255), 2)
    cv2.line(frame, (int(w*0.95), bottom_y), (int(w*0.65), far_y), (0, 255, 255), 2)
    
    cv2.imwrite(output_path, frame)
    cap.release()
    print(f"Frame saved to {output_path}")

if __name__ == "__main__":
    capture_and_overlay("bus.mp4", 750, "line_debug_750.jpg")
