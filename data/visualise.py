import numpy as np

# def read_dataset(filepath):
  
#     x = []
#     room = []
#     for line in open(filepath):
#         if line.strip() != "": # handle empty rows in file
#             row = line.strip().split(" ")
#             x.append(list(map(float, row[:-1]))) 
#             room.append(row[-1])
    
#     x = np.array(x)
#     [classes, y] = np.unique(room, return_inverse=True) 

#     x = np.array(x)
#     y = np.array(y)
#     return (x, y, classes)

# (x, y, classes) = read_dataset("data/clean_dataset.txt")
 
for line in open('clean_dataset.txt'):
    print(line)
    