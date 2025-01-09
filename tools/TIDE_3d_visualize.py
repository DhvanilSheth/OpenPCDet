import json
from pathlib import Path
import matplotlib.pyplot as plt

color_scheme = {
    'missed': '#FF9999',          # Light red
    'false_positive': '#9999FF',     # Light blue
    'localization': '#99FF99', # Light green
    'classification': '#FFCC99', # Light orange
    'both': '#CC99FF',         # Lavender
    'duplicate': '#66CCCC',    # Teal
    'true_positive': '#CCCCCC'       # Gray
}

def plot_error_distribution(error_metrics, save_path):
    """
    Plots a bar chart for error distribution based on the error metrics.

    Args:
        error_metrics (dict): Dictionary containing counts of each error type.
        save_path (str or Path): Path to save the generated plot.
    """
    labels = list(error_metrics.keys())
    values = list(error_metrics.values())
    colors = [color_scheme[label] for label in labels]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color=colors)
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.title('Error Analysis')
    plt.savefig(save_path)
    plt.close()


def plot_error_distribution_pie_chart(error_metrics, save_path):
    """
    Plots a pie chart for error distribution based on the error metrics.

    Args:
        error_metrics (dict): Dictionary containing counts of each error type.
        save_path (str or Path): Path to save the generated plot.
    """
    labels = list(error_metrics.keys())
    sizes = list(error_metrics.values())
    colors = [color_scheme[label] for label in labels]

    plt.figure(figsize=(10, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    error_metrics_path = "/home/dhvanil21040/OpenPCDet/output/home/dhvanil21040/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext/default/eval/epoch_1/val/default/error_metrics.json"
    save_path = "/home/dhvanil21040/OpenPCDet/output/home/dhvanil21040/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext/default/eval/epoch_1/val/default/error_distribution.png"

    error_metrics_path = Path(error_metrics_path)
    save_path = Path(save_path)

    if error_metrics_path.exists():
        with open(error_metrics_path, 'r') as f:
            error_metrics = json.load(f)
        plot_error_distribution(error_metrics, save_path)
        plot_error_distribution_pie_chart(error_metrics, save_path.parent / 'error_distribution_pie_chart.png')
        print(f"Error distribution plot saved to {save_path}")
    else:
        print(f"Error metrics file not found: {error_metrics_path}")