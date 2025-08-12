!pip install ultralytics opencv-python pandas tqdm matplotlib
import pandas
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import warnings
from IPython.display import HTML, display

# ***models and comparison***

def run_inference(model_path, video_path, output_video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return 0, 0  # Return 0 for both avg_time and fps if video loading fails

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    times = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model.track(frame, persist=True, tracker="botsort.yaml")
        end_time = time.time()

        times.append(end_time - start_time)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Avoid ZeroDivisionError if times is empty
    if not times:
        print("Warning: No frames processed. Check video file.")
        return 0, 0

    return sum(times) / len(times), 1/(sum(times)/len(times))

video_path = "/content/test1.mp4"
output_dir = Path("output_videos")
output_dir.mkdir(exist_ok=True)

models = {
    "YOLOv10n": "yolov10n.pt",
    "YOLOv11n": "yolo11n.pt",
    "YOLOv12n": "yolo12n.pt"
}

# Process each model
results = {}
for model_name, model_path in models.items():
    output_path = output_dir / f"{model_name}_output.mp4"
    # Pass model_path, video_path, and output_path to run_inference
    avg_time = run_inference(model_path, video_path, str(output_path)) # Changed this line
    results[model_name] = avg_time

# Print comparison
print("\nInference Speed Comparison:")
print("-" * 50)
print(f"{'Model':<15} {'Avg Inference Time (ms)':<25} {'FPS':<10}")
print("-" * 50)
for model_name, (avg_time, fps) in results.items():
    print(f"{model_name:<15} {avg_time * 1000:<25.2f} {fps:<10.2f}")
print("-" * 50)


import matplotlib.pyplot as plt

# Ensure plots show in notebook
%matplotlib inline

# Data extraction
model_names = list(results.keys())
avg_times = [results[m][0] * 1000 for m in model_names]  # Convert seconds to milliseconds
fps_values = [results[m][1] for m in model_names]

# Bar chart for average inference time
plt.figure(figsize=(8, 5))
plt.bar(model_names, avg_times, color='skyblue')
plt.title("Average Inference Time per Model")
plt.ylabel("Time (ms)")
plt.xlabel("YOLO Model")
plt.grid(True, axis='y')
plt.show()

# Line chart for FPS
plt.figure(figsize=(8, 5))
plt.plot(model_names, fps_values, marker='o', color='orange')
plt.title("Frames Per Second (FPS) per Model")
plt.ylabel("FPS")
plt.xlabel("YOLO Model")
plt.grid(True, axis='y')
plt.show()

