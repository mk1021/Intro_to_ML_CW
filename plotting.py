
import matplotlib.pyplot as plt


def plot_tree(node, depth=0, x=0, y=20):
    """ Recursively traverse the decision tree and create a graphical
    representation of its nodes and branches

    Args:
        node (dict): A dictionary representing a node in the decision tree.
        depth (int): The depth of the current node in the tree.
        x (int): The x-coordinate of the current node.
        y (int): The y-coordinate of the current node.

    Returns:
        None
    """
    spacing = 10 # Vertical spacing between tree levels

    if 'leaf' in node:
        # If the current node is a leaf node:
        # Display the leaf node
        plt.text(x, y, str(node).strip('{}'), fontsize=10,
                 ha='center', va='center', bbox=dict(facecolor='green'))
    else:
        # Display the split condition
        attribute, split_value = node['split']
        plt.text(x, y, f"x[{attribute}] < {split_value}", fontsize=10,
                 ha='center', va='center', bbox=dict(facecolor='white'))

        child_y_pos = y - spacing
        l_child_pos = x - 100 * (1 / (2 ** depth))
        r_child_pos = x + 100 * (1 / (2 ** depth))

        # Plot lines connecting the current node to its children
        plt.plot([l_child_pos, x, r_child_pos], [child_y_pos, y, child_y_pos],
                 marker='o')

        # Recursively plot the left and right child nodes
        plot_tree(node['left'], depth + 1, l_child_pos, child_y_pos)
        plot_tree(node['right'], depth + 1, r_child_pos, child_y_pos)


def generate_tree(root, max_depth, filename):
    """ Generate a graphical representation of a decision tree and
        save it as an image file.

    Args:
        root (dict): The root node of the decision tree.
        max_depth (int): The maximum depth of the tree.
        filename (str): The filename to save the image.

    Returns:
        None
    """
    # Create a Matplotlib figure with a specific size
    fig = plt.figure(figsize=(max_depth, max_depth), dpi=80)

    # get the graphical representation of the tree
    plot_tree(root)

    # Remove the axes from the plot
    plt.axis('off')

    # Save the figure as filename
    plt.savefig(filename)

    # Close the figure
    plt.close()
