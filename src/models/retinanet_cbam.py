from collections import OrderedDict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
import torch
from src.models.utils._cbam import CBAM
from src.utils import masks_list_to_targets_list

class BackboneWithCBAM(nn.Module):
    # Channel dims for ResNet50 layer2-4 (C3-C5)
    RESNET50_OUT_CHANNELS = OrderedDict({
        "layer2": 512,  # C3 0
        "layer3": 1024, # C4 1
        "layer4": 2048, # C5 2
    })

    def __init__(self, backbone_body, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.body = backbone_body
        self.cbam_modules = nn.ModuleDict({
            layer: CBAM(channels, reduction_ratio, kernel_size)
            for layer, channels in self.RESNET50_OUT_CHANNELS.items()
        })

    def forward(self, x):
        outputs = OrderedDict()
        
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.cbam_modules:
                x = self.cbam_modules[name](x)
                index = list(self.RESNET50_OUT_CHANNELS.keys()).index(name)
                index = str(index)
                outputs[index] = x

        return outputs

class RetinaNetCBAM(nn.Module):
    def __init__(self, device="cpu", state_dict=None):
        super(RetinaNetCBAM, self).__init__()
        self.device = device
        self.model = retinanet_resnet50_fpn_v2(
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=2, 
            trainable_backbone_layers=3
            ).to(self.device)
        
        # swap out the backbone body with the CBAM-wrapped version
        original_body = self.model.backbone.body
        self.model.backbone.body = BackboneWithCBAM(original_body).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        return self.model(images)

    def train_model(self, train_dataloader, lr, epochs, targets_are_masks = True):
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr)

        self.train()

        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (images, targets_list) in enumerate(train_dataloader):
                images = images.to(self.device)

                if targets_are_masks:
                    targets_list = masks_list_to_targets_list(targets_list)
                
                # Move targets (List[Dict]) to the correct device
                targets_list = [{k: v.to(self.device) for k, v in t.items()} for t in targets_list]
                
                loss_dict = self(images, targets=targets_list)
                loss_detection_cls = loss_dict['classification'] # Focal Loss
                loss_detection_box = loss_dict['bbox_regression'] # Smooth L1 Loss
                
                # Total Loss
                loss = loss_detection_cls + loss_detection_box 
                
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_dataloader)

            print(f"--- Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} ---") 

        self.eval()
        print("Training Complete")

    def test_model(self, test_dataloader, targets_are_masks = True):
        evaluator = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

        with torch.inference_mode():
            self.eval()
            for batch_idx, (images, targets_list) in enumerate(test_dataloader):
                images = images.to(self.device)
                
                if targets_are_masks:
                    targets_list = masks_list_to_targets_list(targets_list)
                
                targets_on_device = [{k: v.to(self.device) for k, v in t.items()} for t in targets_list]
                detections = self(images)
                evaluator.update(detections, targets_on_device)
            
            metrics = evaluator.compute()
            return metrics