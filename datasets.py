import os
import torch
import torch.utils.data
import torchvision
import cv2
from pycocotools.coco import COCO


class TAKODataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def define_train(self, train):
        self.train = train

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes, categories = [], []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0

            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            categories.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(categories, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        annotation = {}
        annotation["boxes"] = boxes
        annotation["labels"] = labels
        annotation["image_id"] = img_id
        annotation["area"] = areas
        annotation["iscrowd"] = iscrowd

        return img, annotation

    def __len__(self):
        return len(self.ids)


class WrapperDataset(torch.utils.data.Dataset):

    def __init__(self, subset, to_tensor_transform, aug_transform=None):
        self.subset = subset
        self.aug_transform = aug_transform
        self.to_tensor_transform = to_tensor_transform()

    def __getitem__(self, index):
        img, annotation = self.subset[index]

        if self.aug_transform:
            boxes = annotation["boxes"].tolist()
            labels = annotation["labels"].tolist()
            try:
                transformed = self.aug_transform()(image=img, bboxes=boxes, category_ids=labels)

                img = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['category_ids']
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)

                annotation["boxes"] = boxes
                annotation["labels"] = labels
            except Exception as exc:
                print(exc, f"Invalid transformation of box: {boxes}")

        img = self.to_tensor_transform(img)
        return img, annotation

    def __len__(self):
        return len(self.subset)
