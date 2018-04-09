def convert_with_cairosvg_sizes(width,height,file,outfile):
    from cairosvg.surface import PNGSurface
    with open(file, 'rb') as svg_file:
        PNGSurface.convert(
            bytestring=svg_file.read(),
            width=width,
            height=height,
            write_to=open(outfile, 'wb')
            )

import os
file_annotations = open(os.getcwd() +"/stroke_annotations.csv","r")
for f in file_annotations:
    filename = f.split(",")[0]
    outfile = filename.split(".")[0] + ".png"
    filename = os.getcwd()+"/left_!/stroke/"+filename
    outfile = os.getcwd()+"/left_!/stroke_png/"+outfile
    print(filename,outfile)
    convert_with_cairosvg_sizes(200,200,filename,outfile)
