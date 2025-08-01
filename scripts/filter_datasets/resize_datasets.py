

# Resize al img from SOTAdatasets to 40x40

import cv2, os




SOTA_datasets = [
    "cifar100_SOTA_testSet",
    "cifar100_SOTA_trainSet",
    "miniImageNet_SOTA_testSet",
    "miniImageNet_SOTA_trainSet",
    "omniglot_SOTA_testSet",
    "omniglot_SOTA_trainSet",
    "organamnist"
]

resize_resolution = (40, 40)

def resize_image(dir_path):
    for img_filename in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_filename)
        if img_filename.endswith(".jpg") or img_filename.endswith(".png") or img_filename.endswith(".JPEG") or img_filename.endswith(".JPG"):
            # Read the image
            image = cv2.imread(img_path)
            if image is not None:
                # Resize the image
                image = cv2.resize(
                    image, resize_resolution, interpolation=cv2.INTER_AREA
                )  # Resize
                # Overwrite the image with the resized one
                cv2.imwrite(img_path, image)


# Loop through each dataset and resize images
for dataset in SOTA_datasets:
    # Load the dataset  
    dataset_path = f"datasets/{dataset}/data/"
    # Loop through each directory in the dataset

    for dir_filename in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, dir_filename)
        if os.path.isdir(dir_path):
            resize_image(dir_path)
            for dir_filename_ in os.listdir(dir_path):
                dir_path_ = os.path.join(dir_path, dir_filename_)
                if os.path.isdir(dir_path_):
                    resize_image(dir_path_)
                    