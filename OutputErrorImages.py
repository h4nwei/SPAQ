### OutputErrorImages.py
### This file is to Test which image is be demaged, and print their name in console. You should copy the output and paste them at your local device as "demagedImgs.txt" and run 'python pickImage.py'.

import os
import os.path
from openpyxl import load_workbook
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import scipy.io
import xlrd
from tqdm import tqdm

## DEFINE Args
root = "" # your SPAQ dataset location
index = list(range(0, 11125))
transform = transforms = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])

## START loading SPAQ dataset
data = xlrd.open_workbook(os.path.join(root, 'Annotations/MOS and Image attribute scores.xlsx'))
table = data.sheet_by_index(0)

for rowNum in tqdm(range(table.nrows)):
    if rowNum > 0:
        rowValue = table.row_values(rowNum)
        # acquire imgnames, convert them into RGB. If OSErr was throw, then print their name for record.
        try:
            sample = pil_loader(os.path.join(root, 'TestImage', rowValue[0]))
        except OSError as e:
            print(os.path.join(root, 'TestImage', rowValue[0]))
print("Success")
