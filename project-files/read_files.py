import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

file_annotations = open(os.getcwd() +"/path_annotations.csv","r")
for f in file_annotations:
    filename = f.split(",")[0]
    if len(f.split(",")) > 2:
        print(filename)
        continue
    if len(filename.split(".")) <= 1 :
        continue
    filename = os.getcwd()+"/left_!/path_png/"+filename
    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.title(f.split(",")[0])
    plt.show()

