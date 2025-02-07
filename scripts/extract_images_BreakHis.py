## Apply transformations and join all images by classes in data folder

import os
from PIL import Image






class Ds:

    def __init__(self):

        self.src_dataset = "datasets/BreaKHis_v1/histology_slides/breast"
        self.tgt_dataset = "datasets/BreaKHis_formatted/data"
        os.makedirs(os.path.dirname(self.tgt_dataset), exist_ok=True)


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

    # If folder, add name to class_name and continue iterating. If file, read it and save it in the target folder
    def add_class(self, tgt_folder, src_folder, class_name, prefix_namefile = ""):
        for f in os.listdir(src_folder):
            src = src_folder + "/" + f
            if os.path.isdir(src):
                new_pref = prefix_namefile + f + "__" if prefix_namefile != "" else f + "__"
                self.add_class(tgt_folder, src, class_name, new_pref)
            else:
                tgt_img = tgt_folder + "/" + class_name + "/" + prefix_namefile + f
                os.makedirs(os.path.dirname(tgt_img), exist_ok=True)
                self.read_save_image(src, tgt_img, 0)


    ### Main function
    def main(self):
        # Remove target folder
        if os.path.exists(self.tgt_dataset):
            os.system("rm -r " + self.tgt_dataset)
        os.makedirs(self.tgt_dataset, exist_ok=True)

        ## For each line, apply transformations and save in the target folder
        for b_folder in ['benign', 'malignant']:
            st_folder = self.src_dataset + "/" + b_folder + "/SOB/"
            
            for class_ in os.listdir(st_folder):
                cl_folder = st_folder + class_
                print(cl_folder, b_folder + "__" + class_)
                self.add_class(tgt_folder=self.tgt_dataset, src_folder=cl_folder, class_name=b_folder + "__" + class_)


if __name__ == "__main__":
    D = Ds()
    D.main()



    
    