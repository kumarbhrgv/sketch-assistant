import os

image_path = os.getcwd() + "/left_!/path_png/"
file_annotations = open(os.getcwd() +"/path_annotations.csv","w")

lst = os.listdir(image_path)
lst.sort()

for f in lst:
    filename = os.fsdecode(f)
    file_annotations.write(filename+"," + os.linesep)

image_stroke = os.getcwd() + "/left_!/stroke_png/"
file_annotations.close()
file_annotations = open(os.getcwd() +"/stroke_annotations.csv","w")
lst = os.listdir(image_stroke)
lst.sort()

for f in lst:
    filename = os.fsdecode(f)
    file_annotations.write(filename+"," + os.linesep)
file_annotations.close()