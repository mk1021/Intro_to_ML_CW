import numpy as np
import matplotlib.pyplot as plt


def read_dataset(filepath):
    x = []
    room = []
    for line in open(filepath):
        if line.strip() != "":  # handle empty rows in file
            row = line.strip().split("\t")
            x.append(list(map(float, row[:-1])))
            room.append(row[-1])

    [classes, y] = np.unique(room, return_inverse=True)

    x = np.array(x)
    y = np.array(y)

    return (x, y, classes)


(x, y, classes) = read_dataset("clean_dataset.txt")

for i in range(0, 4):
    print("Class " + str(i))
    print(x[y == i].min(axis=0))
    print(x[y == i].max(axis=0))
    print(np.median(x[y == i], axis=0))

# feature_names = ["WIFI_1", "WIFI_2", "WIFI_3", "WIFI_4", "WIFI_5", "WIFI_6",
#                  "WIFI_7"]
#
# # plt.figure()
# # ax = plt.axes(projection='3d')
# #plt.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k')
# # plt.xlabel(feature_names[0])
# # plt.ylabel(feature_names[1])
# # plt.show()


fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(x[:, 2], x[:, 3], x[:, 4], c=y, cmap=plt.cm.Set1, edgecolor='k')
ax.set_title('3D Scatter Plot')
plt.show()