from PIL import Image

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

import torchvision
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn, MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import os
import importlib
import sys
from pathlib import Path

import utils
import train
from train import train_one_epoch, evaluate
import transforms as T

# class MaskRCNNWrapper():

#     def __init__(self, num_classes, pretrained=False, load_from_path=None):
       
#         if load_from_path:
#             self.model = maskrcnn_resnet50_fpn()
#             self.model.load_state_dict(torch.load(load_from_path))
#         elif pretrained:
#             weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
#             self.model = maskrcnn_resnet50_fpn(weights=weights)
#         else:
#             self.model = maskrcnn_resnet50_fpn(weights=None)

#             # get number of input features for the classifier
#         in_features = self.model.roi_heads.box_predictor.cls_score.in_features
#         # replace the pre-trained head with a new one
#         self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#         # now get the number of input features for the mask classifier
#         in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
#         hidden_layer = 256
#         # and replace the mask predictor with a new one
#         self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
#                                                                 hidden_layer,
#                                                                 num_classes)
    
#     def do_training(self, 
#                     torch_dataset, 
#                     torch_dataset_test, 
#                     num_epochs=4,
#                     lr=0.005,
#                     momentum=0.5,
#                     weight_decay=0.0005,
#                     step_size=20,
#                     print_freq=10):
        
#         data_loader = torch.utils.data.DataLoader(
#             torch_dataset, batch_size=4, shuffle=False, num_workers=8,
#             collate_fn=utils.collate_fn)
#         data_loader_test = torch.utils.data.DataLoader(
#             torch_dataset_test, batch_size=2, shuffle=False, num_workers=8,
#             collate_fn=utils.collate_fn)
        
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"Using {device}")
#         self.model.to(device)

#         params = [p for p in self.model.parameters() if p.requires_grad]
#         optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

#         lr_scheduler = torch.optim.lr_scheduler(optimizer, step_size=step_size, gamma=0.1)
        
#         for epoch in range(num_epochs):
#             train_one_epoch(model=self.model,
#                             optimizer=optimizer,
#                             data_loader=data_loader,
#                             device=device,
#                             epoch=epoch,
#                             print_freq=print_freq)
            
#             lr_scheduler.step()
#             evaluate(model=self.model, data_loader=data_loader_test, device=device)
        
#     def save_model(self, file_name):
#         torch.save(self.model.state_dict(), os.path.abspath(f"maskrcnn_weights/{file_name}"))

def do_training(model, 
                torch_dataset, 
                torch_dataset_test, 
                num_epochs=4,
                lr=0.005,
                momentum=0.5,
                weight_decay=0.0005,
                step_size=20,
                print_freq=10):
    
    data_loader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=4, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        torch_dataset_test, batch_size=4, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    for epoch in range(num_epochs):
        train_one_epoch(model=model,
                        optimizer=optimizer,
                        data_loader=data_loader,
                        device=device,
                        epoch=epoch,
                        print_freq=print_freq)
        
        lr_scheduler.step()
        evaluate(model=model, data_loader=data_loader_test, device=device)
    
def save_model(model, file_name):
    torch.save(model.state_dict(), os.path.abspath(f"maskrcnn_weights/{file_name}"))

def get_model(num_classes, pretrained=True):
    if pretrained:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model