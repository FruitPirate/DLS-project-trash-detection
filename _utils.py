import colorsys
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ExifTags
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from torchvision import transforms


def find_image_id(image_filepath, images_list):
    img_id = -1
    for img in images_list:
        if img['file_name'] == image_filepath:
            return img['id']

    if img_id == -1:
        return None


def print_image(image_filepath,
                dataset_path,
                images,
                coco,
                cat_names,
                super_cat_names,
                cat_ids_2_super_cat_ids,
                text=False):

    img_id = find_image_id(image_filepath, images)

    full_path = os.path.join(dataset_path, image_filepath)

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    if img_id is None:
        print('Incorrect file name')
        return None
    else:
        I = Image.open(full_path)

        if I._getexif():
            exif = dict(I._getexif().items())
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180, expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270, expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90, expand=True)

        # Show image
        fig, ax = plt.subplots(1)
        plt.axis('off')
        plt.imshow(I)

        # Load mask ids
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[], iscrowd=None)
        anns_sel = coco.loadAnns(annIds)

        # Show annotations
        for ann in anns_sel:
            color = colorsys.hsv_to_rgb(np.random.random(), 1, 1)
            for seg in ann['segmentation']:
                poly = Polygon(np.array(seg).reshape((int(len(seg)/2), 2)))
                p = PatchCollection([poly], facecolor=color, edgecolors=color,linewidths=0, alpha=0.4)
                ax.add_collection(p)
                p = PatchCollection([poly], facecolor='none', edgecolors=color, linewidths=2)
                ax.add_collection(p)
            [left, bottom, width, height] = ann['bbox']
            rect = Rectangle((left, bottom), width, height, linewidth=2, edgecolor=color,
                             facecolor='none', alpha=0.7, linestyle = '-')
            ax.add_patch(rect)

            if text:
                right = left + width
                top = bottom + height
                cat_name = cat_names[ann['category_id']]
                super_category = super_cat_names[cat_ids_2_super_cat_ids[ann['category_id']]]

                ax.text(right, top,
                        f"supercategory: {super_category}\ncategory: {cat_name}",
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=20,
                        color=color)

        plt.show()


def print_out_image(image, annotation, mapping, cut_off=0.3):

    I = transforms.ToPILImage()(image.cpu().detach()).convert("RGB")

    # Show image
    fig, ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(I)

    # Show annotations

    boxes = annotation['boxes']
    labels = annotation['labels']
    scores = annotation['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score < cut_off:
            continue
        color = colorsys.hsv_to_rgb(np.random.random(), 1, 1)
        [left, top, right, bot] = box.cpu().detach().numpy()
        width = right - left
        height = bot - top
        rect = Rectangle((left, top), width, height, linewidth=2, edgecolor=color,
                         facecolor='none', alpha=0.7, linestyle = '-')
        ax.add_patch(rect)

        ax.text(right, bot,
                f"{mapping[label.item()]} | {score:.2f}",
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=20,
                color=color)

    plt.show()
