import numpy as np
import matplotlib.pyplot as plt
from decision_tree import decision_tree_learning

# Define a simple decision tree structure (you should replace this with your actual tree)
def plot_tree(node, depth, x, y):
    x = 0
    y = 100
    spacing = 10
    
    if 'leaf' in node:
        # plt.plot(x, y)
        plt.text(x, y, node, fontsize = 10, 
                 bbox = dict(facecolor = 'green', alpha = 0.5))
    else:
        attribute, split_value = node['split']
        plt.text(x, y, f"x[{attribute}] < {split_value}", fontsize = 10, 
                 bbox = dict(facecolor = 'white', alpha = 0.5))
        
    child_y_pos = y - spacing
    l_child_pos = x - 1/(2**depth)
    r_child_pos = x + 1/(2**depth)
    plt.plot([l_child_pos, x, r_child_pos], [child_y_pos, y, child_y_pos])
    
    plot_tree(node['left'], depth+1, l_child_pos, child_y_pos)
    plot_tree(node['right'], depth+1, r_child_pos, child_y_pos)
        
def generate_tree(node, maxdepth, filename):
    
    