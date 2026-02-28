import os
# AGGRESSIVELY disable oneDNN/MKLDNN to prevent "OneDnnContext does not have the input Filter" crash on Windows
os.environ['PADDLE_ONEDNN_DISABLE'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_mkldnn_bfloat16'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from processor import VideoUploadProcessor
import time

def progress_callback(pct):
    print(f"Progress: {pct}%")

def verify():
    video_path = "bus.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    print(f"Starting analysis for {video_path}...")
    # Initialize processor with default paths
    processor = VideoUploadProcessor()
    
    start_time = time.time()
    processor.process_video(video_path, progress_callback=progress_callback)
    end_time = time.time()
    
    print(f"Analysis complete in {end_time - start_time:.2f}s")
    print("Check terminal/database for logged events.")

if __name__ == "__main__":
    # Ensure there is a dummy mongo connection or use memory
    verify()
