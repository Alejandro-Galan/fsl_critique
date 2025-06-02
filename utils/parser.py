import json
import os
from typing import Tuple

import cv2
import numpy as np
import torch
import datasets.config as config


from utils.preprocessing import preprocess_image
from utils.constants import Const_c

def parse_files(ds_name: str, files_set) -> Tuple[list, list]:
    # Set global variables
    config.set_data_dirs(ds_name=ds_name)


    if not files_set:
        # Retrieve filepaths
        img_filenames = [
            fname
            for fname in os.listdir(config.images_dir)
            if fname.endswith(config.image_extn)
        ]
    else:
        img_filenames = files_set["images"]

    # Parse files
    if "json" in config.label_extn:
        images, labels = parse_files_json(img_filenames=img_filenames)
    else:
        images, labels = parse_files_txt(img_filenames=img_filenames)

    return images, labels

def extract_image_label(label_name, char, doc_path):
    
    char_path = os.path.join(doc_path, char)
    # Retrieve filepaths
    img_filenames = [
        os.path.join(char_path, fname)
        for fname in os.listdir(char_path)
        if fname.endswith(config.image_extn)
    ]
    return img_filenames, [label_name] * len(img_filenames)



## Doing Omniglot, miniImageNet not prepared yet
def parse_files_SOTA(ds_name: str, files_set) -> Tuple[list, list]:
    # Set global variables
    config.set_data_dirs(ds_name=ds_name)

    # Retrieve a subset
    if files_set:
        all_filepaths, all_labels = files_set["images"], files_set["labels"]
    else:
        all_filepaths, all_labels = [], []
        for doc in os.listdir(config.images_dir):
            doc_path = os.path.join(config.images_dir, doc)
            if ds_name.startswith("miniImageNet") or ds_name.startswith(Const_c.DATASETS.BREAKHIS.value):
                images, labels = extract_image_label(label_name=doc, char=doc, doc_path=config.images_dir)
                all_labels += labels
                all_filepaths += images
            elif ds_name.startswith("omniglot") :
                for char in os.listdir(doc_path):
                    label_name = doc + "_" + char
                    images, labels = extract_image_label(label_name, char, doc_path)
                    all_labels += labels
                    all_filepaths += images

    # Extract images (whole image as bbox)
    images = parse_files_no_txt_SOTA(img_filenames=all_filepaths)

    return images, all_labels


def load_img_pages() -> list:
    # Retrieve filepaths
    img_filenames = [
        fname
        for fname in os.listdir(config.images_dir)
        if fname.endswith(config.image_extn)
    ]

    # Parse images and preprocess them
    images = []
    for img_filename in img_filenames:
        image_path = config.images_dir / img_filename
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            image = preprocess_image(image, resize=False)
            images.append(image)
    return images


####################################################################### JSON FILES [MUSIC DATASETS]:


def parse_files_json(
    img_filenames: list, return_position: bool = False
) -> Tuple[list, list]:
    bboxes = []
    glyphs = []
    positions = []

    for img_filename in img_filenames:
        label_path = config.labels_dir / f"{img_filename + config.label_extn}"
        image_path = config.images_dir / img_filename

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            with open(label_path) as json_file:
                data = json.load(json_file)
                if "pages" in data:
                    for page in data["pages"]:
                        if "regions" in page:
                            for region in page["regions"]:
                                if region["type"] == "staff" and "symbols" in region:
                                    symbols = region["symbols"]
                                    if len(symbols) > 0:
                                        if all(
                                            [
                                                False
                                                for s in symbols
                                                if "bounding_box" not in s.keys()
                                            ]
                                        ):
                                            symbols.sort(
                                                key=lambda symbol: symbol[
                                                    "bounding_box"
                                                ]["fromX"]
                                            )
                                            r_top, r_left, r_bottom, r_right = (
                                                region["bounding_box"]["fromY"],
                                                region["bounding_box"]["fromX"],
                                                region["bounding_box"]["toY"],
                                                region["bounding_box"]["toX"],
                                            )
                                            if (
                                                image[r_top:r_bottom, r_left:r_right]
                                                is not None
                                            ):
                                                for s in symbols:
                                                    if (
                                                        "bounding_box" in s
                                                        and "agnostic_symbol_type" in s
                                                        and "position_in_staff" in s
                                                    ):
                                                        (
                                                            s_top,
                                                            s_left,
                                                            s_bottom,
                                                            s_right,
                                                        ) = (
                                                            s["bounding_box"]["fromY"],
                                                            s["bounding_box"]["fromX"],
                                                            s["bounding_box"]["toY"],
                                                            s["bounding_box"]["toX"],
                                                        )

                                                        if (
                                                            (s_bottom - s_top) != 0
                                                            and (s_right - s_left) != 0
                                                            and (r_bottom - r_top) != 0
                                                        ):
                                                            bboxes.append(
                                                                preprocess_image(
                                                                    image[
                                                                        s_top:s_bottom,
                                                                        s_left:s_right,
                                                                    ]
                                                                )
                                                            )
                                                            glyphs.append(
                                                                s[
                                                                    "agnostic_symbol_type"
                                                                ]
                                                            )
                                                            positions.append(
                                                                s["position_in_staff"]
                                                            )

    if return_position:
        return bboxes, positions

    return bboxes, glyphs


####################################################################### TXT FILES [TEXT DATASETS]:


def parse_files_txt(img_filenames: list) -> Tuple[list, list]:
    bboxes = []
    glyphs = []

    for img_filename in img_filenames:
        label_filename = img_filename.replace(config.image_extn, config.label_extn)
        label_path = config.labels_dir / label_filename

        image_path = config.images_dir / img_filename
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            with open(label_path) as f:
                line = f.readlines()

            for line in line:
                line_s = line.split()
                symbol = line_s[0]
                y_s, x_s, y_l, x_l = [int(float(u)) for u in line_s[1:]]

                if x_l > x_s and y_l > y_s:
                    bboxes.append(preprocess_image(image[x_s:x_l, y_s:y_l]))
                    glyphs.append(symbol)

    return bboxes, glyphs


def parse_files_no_txt_SOTA(img_filenames: list) -> Tuple[list, list]:
    bboxes = []

    for image_path in img_filenames:
        
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            breakpoint()
        if image is not None:
            bboxes.append(preprocess_image(image))

    return bboxes
