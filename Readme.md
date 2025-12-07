VisualiseAI

Author: Brendon Worldon + ChatGPT
Status: Demo-ready, polished for presentations and GitHub

Overview

VisualiseAI demonstrates YOLOv8 object detection and analytics in a safe, offline demo. It’s designed for presentations, portfolio showcases, or resume highlights.

Note: This project focuses on inference only. The train/val folders are placeholders for potential future training.

Key features:

Pretrained YOLOv8 detection

Confidence filtering to remove false detections

Annotated image outputs

CSV reporting and analytics

Polished demo slide with charts

Features

Single-image detection with confidence-labeled bounding boxes

Batch folder detection via detect_folder.py

CSV reports:

results_demo/detection_report.csv – full detection details

results_demo/detection_summary.csv – per-image summary

Visual analytics:

Pie chart of class distribution

Stacked bar chart of detections per image