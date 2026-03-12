# src/plot_utils.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_class_distribution(class_counts):
    """
    Plots a pie chart of total class counts.
    
    Parameters:
        class_counts (pd.Series): Total count per class.
    """
    if class_counts.empty:
        print("Warning: class_counts is empty.")
        return
    class_counts.plot.pie(
        autopct='%1.1f%%', 
        title="Detection Class Distribution",
        figsize=(6,6)
    )
    plt.ylabel("")  
    plt.show()


def plot_per_image_summary(per_image_summary):
    """
    Plots a stacked bar chart of per-image detection counts.
    
    Parameters:
        per_image_summary (pd.DataFrame): Each row = image, each column = class count.
    """
    if per_image_summary.empty:
        print("Warning: per_image_summary is empty.")
        return
    per_image_summary.plot(
        kind='bar', 
        stacked=True, 
        figsize=(8,5), 
        title="Per-Image Detection Summary"
    )
    plt.xlabel("Images")
    plt.ylabel("Counts")
    plt.show()

if __name__ == "__main__":
    # Dummy data for testing
    class_counts = pd.Series([50, 30, 20], index=["cat", "dog", "bird"])
    per_image_summary = pd.DataFrame({
        "cat": [10, 5, 8],
        "dog": [3, 7, 2],
        "bird": [0, 1, 2]
    }, index=["img1", "img2", "img3"])

    plot_class_distribution(class_counts)
    plot_per_image_summary(per_image_summary)
