from PIL import Image
import os


root = "D://images//figures//noise"
for path in os.listdir(root):
    if path.endswith("png"):
        im1 = Image.open(os.path.join(root,path))
        im1 = im1.convert("RGB")
        im1.save(os.path.join(root,path)[:-4]+".jpg")
