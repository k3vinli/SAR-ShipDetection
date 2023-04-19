import fiftyone as fo
import fiftyone.utils.coco as fouc
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image

class HRSIDDataset(Dataset):
    """A class to construct a Pytorch dataset from a FiftyOne dataset
    """
    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="segmentations",
        classes=None):

        self.samples = fiftyone_dataset
        self.transforms = transforms 
        self.gt_field = gt_field
        
        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )
        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        # maps values to individual classes
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}            

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        frame_size = (metadata['width'], metadata['height'])
        print("frame_size", frame_size)
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        masks = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det, metadata, category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)
            masks.append(det.to_segmentation(frame_size=frame_size).mask)
            # masks.append(det.mask)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        masks=torch.as_tensor(masks, dtype=torch.float)
        print("mask shape", masks.shape)
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes