import fiftyone as fo
import fiftyone.utils.coco as fouc
from fiftyone.core.labels import Detection
from PIL import Image

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt
def convert_torch_predictions_bbox(preds, det_id, s_id, w, h, classes, conf_threshold):
    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    for bbox, label, score in zip(
        preds["boxes"].cpu().detach().numpy(), 
        preds["labels"].cpu().detach().numpy(), 
        preds["scores"].cpu().detach().numpy(),
    ):
        if conf_threshold != None:
            if score < conf_threshold:
                continue
        # Parse prediction into FiftyOne Detection object
        x0,y0,x1,y1 = bbox
        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1-x0, y1-y0])
        det = coco_obj.to_detection((w,h), classes)
        det["confidence"] = float(score)
        dets.append(det)
        det_id += 1
        
    detections = fo.Detections(detections=dets)

    return detections, det_id

def add_detections(model, torch_dataset, view, field_name="predictions", conf_threshold=None):
    # Run inference on a dataset and add results to FiftyOne
    torch.set_num_threads(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    model.eval()
    model.to(device)
    image_paths = torch_dataset.img_paths
    classes = torch_dataset.classes
    det_id = 0
    
    with fo.ProgressBar() as pb:
        for img, targets in pb(torch_dataset):
            # Get FiftyOne sample indexed by unique image filepath
            img_id = int(targets["image_id"][0])
            img_path = image_paths[img_id]
            sample = view[img_path]
            s_id = sample.id
            w = sample.metadata["width"]
            h = sample.metadata["height"]
            
            # Inference
            preds = model(img.unsqueeze(0).to(device))[0]
            
            detections, det_id = convert_torch_predictions_bbox(
                preds, 
                det_id, 
                s_id, 
                w, 
                h, 
                classes,
                conf_threshold=conf_threshold
            )
            
            sample[field_name] = detections
            sample.save()


def add_segmentations(model, torch_dataset, view, field_name="predictions", seg_field_name="predictions", conf_threshold=None):
    # Run inference on a dataset and add results to FiftyOne
    torch.set_num_threads(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device %s" % device)

    model.eval()
    model.to(device)
    image_paths = torch_dataset.img_paths
    classes = torch_dataset.classes
    det_id = 0
    
    with fo.ProgressBar() as pb:
        for img, targets in pb(torch_dataset):
            # Get FiftyOne sample indexed by unique image filepath
            img_id = int(targets["image_id"][0])
            img_path = image_paths[img_id]
            sample = view[img_path]
            s_id = sample.id
            w = sample.metadata["width"]
            h = sample.metadata["height"]
            
            # Inference
            preds = model(img.unsqueeze(0).to(device))[0]
            
            detections, segmentations, det_id = convert_torch_predictions_seg(
                preds, 
                det_id, 
                s_id, 
                w, 
                h, 
                classes,
                conf_threshold=conf_threshold
            )
            
            sample[field_name] = detections
            sample[seg_field_name] = segmentations
            sample.save()

def convert_torch_predictions_seg(preds, det_id, s_id, w, h, classes, conf_threshold):
    # Convert the outputs of the torch model into a FiftyOne Detections object
    dets = []
    segs = []
    for bbox, label, score, mask in zip(
        preds["boxes"].cpu().detach().numpy(), 
        preds["labels"].cpu().detach().numpy(), 
        preds["scores"].cpu().detach().numpy(),
        preds["masks"].cpu().detach().numpy()
    ):
        # only add detections if over a specified confidence level
        if conf_threshold != None:
            if score < conf_threshold:
                continue

        # Parse prediction into FiftyOne Detection object
        mask = np.ceil(mask).astype(bool)
        x0,y0,x1,y1 = bbox
        
        # removes predictions that are just a line
        if (x1-x0) < 1 or (y1 - y0) < 1:
            continue
        coco_obj = fouc.COCOObject(det_id, s_id, int(label), [x0, y0, x1-x0, y1-y0])
        det = coco_obj.to_detection((w,h), classes)
        det["confidence"] = float(score)

        # bounding box not passed to seg b/c the bounding box is created from the segmentation
        seg = fo.Detection.from_mask(
            label=classes[label],
            mask=mask[0],
            confidence=float(score),
            id=s_id
        )
        dets.append(det)
        segs.append(seg)
        det_id += 1
        
    detections = fo.Detections(detections=dets)
    segmentations = fo.Detections(detections=segs)
    return detections, segmentations, det_id