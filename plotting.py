import numpy as np
import matplotlib.pyplot as plt


# Define a simple decision tree structure (you should replace this with your actual tree)
def plot_tree(node, depth=0, x=0, y=20):
    spacing = 10

    if 'leaf' in node:
        # plt.plot(x, y)
        plt.text(x, y, str(node).strip('{}'), fontsize=10, ha='center', va='center',
                 bbox=dict(facecolor='green'))
    else:
        attribute, split_value = node['split']
        plt.text(x, y, f"x[{attribute}] < {split_value}",
                 fontsize=10, ha='center', va='center',
                 bbox=dict(facecolor='white'))

        child_y_pos = y - spacing
        l_child_pos = x - 100 * (1 / (2 ** depth))
        r_child_pos = x + 100 * (1 / (2 ** depth))
        plt.plot([l_child_pos, x, r_child_pos], [child_y_pos, y, child_y_pos],
                 marker='o')

        plot_tree(node['left'], depth + 1, l_child_pos, child_y_pos)
        plot_tree(node['right'], depth + 1, r_child_pos, child_y_pos)


def generate_tree(root, max_depth, filename):
    # Create a Matplotlib figure with a specific size
    fig = plt.figure(figsize=(max_depth, max_depth), dpi=80)

    # Assuming you have a function called PlotDecisionTree that takes care of
    # plotting your decision tree. You should replace this with your actual
    # function call.
    plot_tree(root)

    # Remove the axes from the plot
    plt.axis('off')

    # Save the figure as Filename
    plt.savefig(filename)

    # Close the figure
    plt.close()
