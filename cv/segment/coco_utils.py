import copy
import os

import numpy as np
import oneflow as flow
import oneflow.utils.data
import flowvision
from PIL import Image
from pycocotools import mask as coco_mask
from transforms import Compose

class FilterAndRemapCocoCategories:
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = flow.as_tensor(mask, dtype=flow.uint8)
        mask = mask.any(dim=2).type(flow.uint8)
        masks.append(mask)
    if masks:
        masks = flow.stack(masks, dim=0)
    else:
        masks = flow.zeros((0, height, width), dtype=flow.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = flow.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target = (masks * cats[:, None, None]).amax(dim=0).type(flow.int8)
            target[(masks.sum(0) > 1)] = 255
        else:
            target = flow.zeros((h, w), dtype=flow.uint8)
#        print("target shape:",np.unique(target.numpy()))
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = flow.utils.data.Subset(dataset, ids)
    return dataset


def get_coco(root, image_set, transforms):
    PATHS = {
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    # The 2 "Compose" below achieve the same thing: converting coco detection
    # samples into segmentation-compatible samples. They just do it with
    # slightly different implementations. We could refactor and unify, but
    # keeping them separate helps keeping the v2 version clean

    transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), transforms])
    dataset = flowvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset
    
