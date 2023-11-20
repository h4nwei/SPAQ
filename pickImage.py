# This file is to copy the undemaged images into dst dir, for use to re-upload these images.

import os
from PIL import Image
import os.path
import shutil

dst = "./updated/" # image folder which should contains the indamaged images correspponding to the demaged images.
file_name = "demagedImgs.txt"

def openreadtxt(file_name):
    file = open(file_name,'r')  
    file_data = file.readlines() 
    for row in file_data:
        row = row.strip('\n')
        shutil.copy(f"TestImage/{row}", dst)

openreadtxt(file_name)
print("Success")
