import numpy as np
from PIL import Image
import os


data = np.load('datasets/organamnist_64.npz')

sets = ["train", "test", "val"]

# def store_set(set):

## Join all images, as the splits have overlapped classes
all_imgs, all_labels = None, None
for set in sets:
    if not all_imgs is None:
        all_imgs   = np.concatenate((all_imgs,   data[set + "_images"]), axis=0)
        all_labels = np.concatenate((all_labels, data[set + "_labels"]), axis=0)
    else: 
        all_imgs   = data[set + "_images"]
        all_labels = data[set + "_labels"]


output_dir = 'datasets/organamnist/'
output_dir += "data/"

os.system("rm -r " + output_dir)
os.makedirs(output_dir, exist_ok=True)
for i, label in enumerate(all_labels):
    np.testing.assert_equal(len(label), 1)
    name = str(label[0]) + "_class/" + str(i) + "_id"
    img_array = all_imgs[i]


    img = Image.fromarray(img_array)
    new_path = os.path.join(output_dir, f"{name}.png")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    img.save(new_path)

print("Stored images in:", output_dir)
