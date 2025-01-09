
import matplotlib.pyplot as plt

# Define a consistent color scheme for the error categories
color_scheme = {
    'missed_gt': '#FF9999',          # Light red
    'false_positive': '#9999FF',     # Light blue
    'localization_error': '#99FF99', # Light green
    'classification_error': '#FFCC99', # Light orange
    'both_error': '#CC99FF',         # Lavender
    'duplicate_error': '#66CCCC',    # Teal
    'true_positive': '#CCCCCC'       # Gray
}

def plot_pie_chart(errors):
    labels = list(errors.keys())
    sizes = list(errors.values())
    colors = [color_scheme[label] for label in labels]
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def plot_bar_graph(errors):
    labels = list(errors.keys())
    values = list(errors.values())
    colors = [color_scheme[label] for label in labels]
    plt.bar(labels, values, color=colors)
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.title('Error Analysis')
    plt.show()