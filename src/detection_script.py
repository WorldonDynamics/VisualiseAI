"""
YOLOv8 Still-Image Detection Demo (Full Script)
Author: Brendon Worldon + ChatGPT

Features:
- YOLOv8 pretrained detection
- Optional transfer learning / training hooks
- Data augmentation (flip, rotate, color shift, mosaic)
- Hyperparameter tuning (epochs, batch, learning rate)
- Safe batch/single-image detection
- Annotated image saving
- CSV reporting: detection_report.csv, detection_summary.csv
- Visual analytics: pie chart + stacked bar chart

This is a demo-focused, safe, offline project.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ------------------------------------------
# 0. Safety & dependency checks
# ------------------------------------------
try:
    import ultralytics
except ImportError:
    print("Error: 'ultralytics' not installed. Run: pip install ultralytics")
    exit(1)


# ------------------------------------------
# 1. Load pretrained YOLOv8 model
# ------------------------------------------
model_name = "yolov8n.pt"   # lightweight + fast for demos
model = YOLO(model_name)
print(f"[INFO] Loaded YOLOv8 model: {model_name}")


# ------------------------------------------
# 2. Optional training configuration
# (only if you want to train on a custom dataset)
# ------------------------------------------
train_config = {
    "epochs": 50,
    "img_size": 640,
    "batch_size": 16,
    "learning_rate": 0.001,
    # Data augmentation settings
    "flip": 0.5,
    "rotate": 15,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "mosaic": 1.0
}

# Uncomment if you want training:
"""
if os.path.exists("dataset.yaml"):
    model.train(
        data="dataset.yaml",
        epochs=train_config["epochs"],
        imgsz=train_config["img_size"],
        batch=train_config["batch_size"],
        lr0=train_config["learning_rate"],
        flip=train_config["flip"],
        rotate=train_config["rotate"],
        hsv_h=train_config["hsv_h"],
        hsv_s=train_config["hsv_s"],
        hsv_v=train_config["hsv_v"],
        mosaic=train_config["mosaic"]
    )
"""


# ------------------------------------------
# 3. Safe result paths
# ------------------------------------------
RESULTS_FOLDER = "results_demo"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_FOLDER, "detection_report.csv")
SUMMARY_CSV_PATH = os.path.join(RESULTS_FOLDER, "detection_summary.csv")


# ------------------------------------------
# 4. Single-image detection
# ------------------------------------------
def detect_single_image(image_path, save_path=RESULTS_FOLDER, csv_path=CSV_PATH):

    if not os.path.exists(image_path) and not image_path.startswith("http"):
        print(f"[ERROR] Image not found: {image_path}")
        return

    results = model(image_path)
    results[0].show()       # show annotated image
    results[0].save(save_path)

    print(f"[INFO] Annotated image saved to: {save_path}")

    detections = results[0].boxes
    data = []

    if detections is not None and len(detections) > 0:
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            data.append({
                "image": os.path.basename(image_path),
                "class": class_name,
                "confidence": conf
            })

    # Append or create CSV
    df_new = pd.DataFrame(data)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] CSV updated: {csv_path}")


# ------------------------------------------
# 5. Batch folder detection
# ------------------------------------------
def detect_images_in_folder(folder_path, save_path=RESULTS_FOLDER, csv_path=CSV_PATH):

    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder does not exist: {folder_path}")
        return

    imgs = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not imgs:
        print("[INFO] No images found in directory.")
        return

    for img_file in imgs:
        full_path = os.path.join(folder_path, img_file)
        detect_single_image(full_path, save_path, csv_path)


# ------------------------------------------
# 6. CSV analytics + charts
# ------------------------------------------
def generate_report(csv_path=CSV_PATH, summary_csv_path=SUMMARY_CSV_PATH):

    if not os.path.exists(csv_path):
        print("[ERROR] No CSV available. Run detection first.")
        return

    df = pd.read_csv(csv_path)

    # ---- per-class totals ----
    class_counts = df["class"].value_counts()
    print("\n[INFO] Total Detections Per Class:")
    print(class_counts)

    plt.figure(figsize=(6, 6))
    class_counts.plot.pie(autopct="%1.1f%%", title="Class Distribution")
    plt.ylabel("")
    plt.show()

    # ---- per-image table ----
    per_image = df.groupby("image")["class"].value_counts().unstack(fill_value=0)
    print("\n[INFO] Per-Image Summary:")
    print(per_image)

    per_image.to_csv(summary_csv_path)
    print(f"[INFO] Summary CSV saved: {summary_csv_path}")

    per_image.plot(kind="bar", stacked=True, figsize=(8, 5), title="Per-Image Detection Summary")
    plt.xlabel("Image")
    plt.ylabel("Detections")
    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 7. Example Demo Trigger
# ------------------------------------------
if __name__ == "__main__":
    # Run detection on ALL images inside test_images/
    detect_images_in_folder("test_images")

    # Generate charts + summary
    generate_report()
BLOCKLIST = ["tie"]

