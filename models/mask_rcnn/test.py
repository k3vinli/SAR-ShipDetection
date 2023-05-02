import fiftyone as fo
import fiftyone.utils.coco as fouc
from PIL import Image

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from train import train_one_epoch, evaluate
import torchvision
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import matplotlib.pyplot as plt
import os
import utils
from DatasetLoaders import HRSIDDataset


# name = "HRSID_train"
# dataset_dir = os.path.abspath("Datasets/HRSID")
# label_path = os.path.abspath("Datasets/HRSID/annotations/train2017.json")
# # The type of the dataset being imported
# dataset_type = fo.types.COCODetectionDataset

# dataset_train = fo.Dataset.from_dir(
#     dataset_dir=dataset_dir,
#     dataset_type=dataset_type,
#     name=name,
#     labels_path=label_path
# )

# paths = dataset_train.values("filepath")
# sample = dataset_train[paths[1]]
# sample.metadata

# detections = sample["detections"].detections
# segmentations = sample["segmentations"].detections
name = "HRSID_test"
if name in fo.list_datasets():
    dataset_test = fo.load_dataset(name)
else:
    dataset_dir = os.path.abspath("ENEE439/Capstone/Datasets/HRSID")
    label_path = os.path.abspath("ENEE439/Capstone/Datasets/HRSID/annotations/test2017.json")
    # The type of the dataset being imported
    dataset_type = fo.types.COCODetectionDataset

    dataset_test = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
        name=name,
        labels_path=label_path
    )
sample = dataset_test.first()
frame_size = (sample.metadata["width"], sample.metadata["height"])
detection = sample["segmentations"]["detections"][0]

segmentation = detection.to_segmentation(frame_size)
full_img_mask = segmentation.mask
print(type(detection))
print("frame size", frame_size)
print("detection:", detection)
print("segmentation:", segmentation)
print("full img", full_img_mask)