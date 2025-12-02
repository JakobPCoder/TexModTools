# Copyright Notice
# - TexModTools
# - First published 2022 - Copyright, Jakob Wapenhensch
# - https://creativecommons.org/licenses/by-nc/4.0/
# - https://creativecommons.org/licenses/by-nc/4.0/legalcode

import os
import cv2
import numpy as np
import pathlib
from wand.image import Image
INPUT_FILE_FORMAT = ".dds"
OUTPUT_FILE_FORMAT = ".png"

def processImage(path, entry): 
    newPath = path.parent / (path.stem + OUTPUT_FILE_FORMAT)   
    cv2.imwrite(str(newPath), cv2.cvtColor(np.array(Image(filename = path)), cv2.COLOR_BGRA2RGBA))
    os.remove(entry)
    print(newPath)  
    return

basePath = os.path.dirname(os.path.realpath(__file__))
with os.scandir(basePath) as dir:
    for entry in dir:
        if entry.is_file():
            path = pathlib.Path(entry)       
            if path.suffix == INPUT_FILE_FORMAT:                           
                processImage(path, entry)
                
