import os
from PIL import Image
from PIL import ImageChops

file_list = os.listdir("Fontimage/chinese")
for i in file_list:
    img = Image.open(os.path.join("Fontimage/chinese", i))
    if not ImageChops.invert(img).getbbox():
        os.remove(os.path.join("Fontimage/chinese", i))
