## Apply transformations and select the correct partition as the SOTA provides

import os
from PIL import Image

## Omniglot. 
# Rotate images x4 and select partition


class Ds:

    def __init__(self, type_ds):
        self.type_ds = type_ds

        if type_ds == "omniglot":
            self.src_dataset = "datasets/omniglot/data"
            self.tgt_dataset = "datasets/omniglot_SOTA_testSet/data"
            self.test_data   = "datasets/partitions_fixed/omniglot_test.txt"
        elif type_ds == "miniImageNet":
            self.src_dataset = "datasets/miniImageNet"
            self.tgt_dataset = "datasets/miniImageNet_SOTA_testSet"
            self.test_data   = "datasets/partitions_fixed/miniImageNet_test.csv"


    def read_save_image(self, src, tgt, rotation_deg):
        # Read image
        im = Image.open(src)
        # Rotate image
        im = im.rotate(rotation_deg)
        # Save image
        im.save(tgt)


    def treat_line_Omniglot(self, line):        
        line = line.strip("\n")
        prefix_line = "/".join(line.split("/")[:-1])
        rotation_str = line.split("/")[-1]
        rotation_deg = int(rotation_str.split("rot")[1])

        src_images = self.src_dataset + "/" + prefix_line
        
        # Get all images in the folder
        for name_img in os.listdir(src_images):
            src_img = src_images + "/" + name_img
            # tgt = tgt_dataset + "/" + prefix_line + "/" + name_img[:-4] + "__" + rotation_str + ".png"
            tgt = self.tgt_dataset + "/" + prefix_line + "__" + rotation_str + "/" + name_img # Add new classes
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            self.read_save_image(src_img, tgt, rotation_deg)

    def treat_line_miniImageNet(self, line):        
        if line.startswith("filename"):
            return
        line = line.split(",")[0]
        prefix_line = line.split("0000")[0]
        num_str = str(int(line.split("0000")[1].replace(".jpg", ""))) # Removing useless zeros
        image_name = prefix_line + "_" + num_str + ".JPEG"

        src_img = self.src_dataset + "/" + prefix_line + "/" + image_name
        tgt_img = self.tgt_dataset + "/" + prefix_line + "/" + image_name
        
        os.makedirs(os.path.dirname(tgt_img), exist_ok=True)
        self.read_save_image(src_img, tgt_img, 0)


    ### Main function
    def main(self):
        # Remove target folder
        if os.path.exists(self.tgt_dataset):
            os.system("rm -r " + self.tgt_dataset)
        os.makedirs(self.tgt_dataset, exist_ok=True)

        ## Read test data file
        with open(self.test_data) as f:
            lines = f.readlines()

        ## For each line, apply transformations and save in the target folder
        for line in lines:
            print(line)
            if self.type_ds == "omniglot":
                self.treat_line_Omniglot(line)
            elif self.type_ds == "miniImageNet":
                self.treat_line_miniImageNet(line)


if __name__ == "__main__":
    D_o = Ds("omniglot")
    D_o.main()
    # D_m = Ds("miniImageNet")
    # D_m.main()

    
    