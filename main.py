import cv2
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model = YOLO("yolov8m.pt")

# Vid Choice
vid = int(input("Enter video number: "))
if vid == 1:
    cap = cv2.VideoCapture("traffic(1).mp4")
elif vid == 2:
    cap = cv2.VideoCapture("traffic(2).mp4")
elif vid == 3:
    cap = cv2.VideoCapture("traffic(3).mp4")
elif vid == 4:
    cap = cv2.VideoCapture("traffic(4).mp4")
elif vid == 5:
    cap = cv2.VideoCapture("traffic(5).mp4")
elif vid == 6:
    cap = cv2.VideoCapture("traffic(6).mp4")
elif vid == 7:
    cap = cv2.VideoCapture("traffic(7).mp4")
else:
    cap = cv2.VideoCapture("traffic(7).mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = 0
data = []

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# Accumulator
heatmap = np.zeros((720, 1280), dtype=np.float32)

print(fps)
while True:
    success, frame = cap.read()

    if not success:
        break

    frame_resized = cv2.resize(frame, (1280, 720))
    results = model.predict([frame_resized], verbose=False)

    vehicle_count = 0

    for result in results:
        boxes = result.boxes
        class_ids = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        coords = boxes.xyxy.tolist()

        # Processing the Objects
        for i in range(len(class_ids)):
            class_name = model.names[int(class_ids[i])]
            confidence = confidences[i]

            print(f"Class Name: {class_name}, Confidence: {confidence}")

            if confidence > 0.25 and class_name in VEHICLE_CLASSES:
                vehicle_count += 1

                x1 = int(coords[i][0])
                y1 = int(coords[i][1])
                x2 = int(coords[i][2])
                y2 = int(coords[i][3])

                # Vehicle Centres
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(heatmap, (cx, cy), 20, 1, -1)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), [0, 0, 0], 2)
                cv2.putText(frame_resized, class_name, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame_resized, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    timestamp = frame_number / fps
    data.append({
        "timestamp": timestamp,
        "frame_number": frame_number,
        "vehicle_count": vehicle_count
    })
    frame_number += 1

    print(f"Vehicles: {vehicle_count}")

    cv2.imshow("traffic", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# Heat Map
heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = heatmap_normalized.astype(np.uint8)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
cv2.imwrite("heatmap.png", heatmap_colored)


# Get one frame for background
video_files = {1: "traffic(1).mp4", 2: "traffic(2).mp4", 3: "traffic(3).mp4",
               4: "traffic(4).mp4", 5: "traffic(5).mp4", 6: "traffic(6).mp4", 7: "traffic(7).mp4"}

cap2 = cv2.VideoCapture(video_files.get(vid, "traffic(7).mp4"))
success2, sample_frame = cap2.read()
print("Frame read success:", success2)

if not success2 or sample_frame is None:
    print("Could not read sample frame — using blank background for heatmap overlay.")
    sample_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
else:
    sample_frame = cv2.resize(sample_frame, (1280, 720))
cap2.release()

# Blend
sample_frame = sample_frame.astype(np.uint8)
overlay = cv2.addWeighted(sample_frame, 0.6, heatmap_colored, 0.5, 0)
cv2.imwrite("heatmap_overlay.png", overlay)
cv2.imshow("Heatmap Overlay", overlay)
print("Sample frame shape:", sample_frame.shape)
print("Heatmap colored shape:", heatmap_colored.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

df = pd.DataFrame(data)

low_threshold = df["vehicle_count"].quantile(0.25)
high_threshold = df["vehicle_count"].quantile(0.75)


def classify_traffic(count):
    if count <= low_threshold:
        return "low traffic"
    elif count <= high_threshold:
        return "medium traffic"
    elif count > high_threshold:
        return "high traffic"

df["density"] = df["vehicle_count"].apply(classify_traffic)

# Stats Analysis
print("Total frames analyzed:", len(df))
print("Average vehicles per frame:", round(df["vehicle_count"].mean(), 2))
print("Peak vehicle count:", df["vehicle_count"].max())
print("Lowest vehicle count:", df["vehicle_count"].min())
print("\nTraffic density breakdown:")
print(df["density"].value_counts())

df.to_csv("traffic_data.csv", index=False)
print(df.head())

# Charts
# Chart 1 — Vehicle count over time
plt.figure(figsize=(12, 4))
plt.plot(df["timestamp"], df["vehicle_count"], color="steelblue", linewidth=1)
plt.title("Vehicle Count Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Vehicles Detected")
plt.tight_layout()
plt.savefig("traffic_over_time.png")
plt.show()

# Chart 2 — Traffic density breakdown
plt.figure(figsize=(6, 4))
df["density"].value_counts().plot(kind="bar", color=["green", "orange", "red"])
plt.title("Traffic Density Distribution")
plt.xlabel("Density Level")
plt.ylabel("Number of Frames")
plt.tight_layout()
plt.savefig("density_distribution.png")
order = ["low traffic", "medium traffic", "high traffic"]
df["density"].value_counts().reindex(order).plot(kind="bar", color=["green", "orange", "red"])
plt.show()

