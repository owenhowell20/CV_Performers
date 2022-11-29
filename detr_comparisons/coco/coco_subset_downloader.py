from pycocotools.coco import COCO
import requests
import sys
import os


if not(os.path.exists("./annotations/")):
    os.system("wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    os.system("unzip annotations_trainval2017.zip")
# coco anotations path
coco = COCO('./annotations/instances_train2017.json')
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['cat'])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)
print("category size: " + str(len(images)))
# if len(images) > 2000:
#     images = images[:1999]   #change this if you want more/less images
if not(os.path.exists("./coco_data/coco_subset/")):
    os.makedirs("./coco_data/coco_subset/")
# Save the images into a local folder
for im in images:
    img_data = requests.get(im['coco_url']).content
    with open('./coco_data/coco_subset/' + im['file_name'], 'wb') as handler:
        handler.write(img_data)