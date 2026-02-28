import cv2
import os

def extract_frames(video_path, output_dir, count=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}, FPS: {fps}, Total Frames: {frame_count}")

    for i in range(count):
        # Sample frames at intervals
        target_frame = int((i / count) * frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            out_name = os.path.join(output_dir, f"frame_{i}.png")
            cv2.imwrite(out_name, frame)
            print(f"Saved {out_name}")
        else:
            print(f"Failed to read frame {target_frame}")
    
    cap.release()

if __name__ == "__main__":
    extract_frames("bus_test_video.mp4", "debug_frames")
