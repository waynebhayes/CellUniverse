import os
from PIL import Image

UNIT_SIZE = 320 
TARGET_WIDTH = 2 * UNIT_SIZE 

path = "PATH TO RAW DATA AND OUTPUT OF SIMULATED ANNEALER"
# the name output of simulated annealer should add "_1" on rawdata eg: "frame000.png" and "frame000_1.png"
images = [] 
for root, dirs, files in os.walk(path):     
    for f in files:
        images.append(f)
print(len(images))
for i in range(len(images)/2): 
    imagefile = []
    j = 0
    for j in range(2):
        imagefile.append(Image.open(path+'/'+images[i*2+j])) 
    target = Image.new('RGB', (TARGET_WIDTH, 240))
    left = 0
    right = UNIT_SIZE
    for image in imagefile:     
        target.paste(image, (left, 0, right, 240))
        left += UNIT_SIZE 
        right += UNIT_SIZE 
        quality_value = 100 
        target.save(path+'/result/'+os.path.splitext(images[i*2+j])[0]+'.jpg', quality = quality_value)
    imagefile = []