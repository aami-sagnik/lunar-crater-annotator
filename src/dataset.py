import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon
import json

MEAN=(0.485, 0.456, 0.406)
STD=(0.229, 0.224, 0.225)

class LunarCraterDataset(Dataset):
    image_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    def __init__(self, n_att, source_h, source_w, dataset_path):
        self.dataset_path = dataset_path
        self.n_att = n_att
        self.source_h = source_h
        self.source_w = source_w
        self.resize_transform = None

        # Collect all PNG image filenames in the dataset directory
        self.png_files = [f for f in os.listdir(dataset_path) if f.endswith(".png") and os.path.isfile(os.path.join(dataset_path, f))]

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return [self[i] for i in indices]

        if isinstance(idx, (list, tuple, np.ndarray, torch.Tensor)):
            return [self[int(i)] for i in idx]

        # Lazy load image from disk
        img_path = os.path.join(self.dataset_path, self.png_files[idx])
        image = Image.open(img_path) 
        image = self.image_transforms(image)

        # Lazy load corresponding masks from JSON
        json_file_base = self.png_files[idx].split(".png")[0]
        masks, num_mask = self.get_masks_from_json(
            os.path.join(self.dataset_path, f"{json_file_base}_annotations.json"),
            self.source_w, self.source_h, self.n_att
        )

        if self.resize_transform is not None:
            image = self.resize_transform(image)
            masks = self.resize_transform(masks)
        
        return image, masks

    def resize(self, h, w):
        self.resize_transform = transforms.Compose([
        transforms.Resize((h, w))
    ])

    def __str__(self):
        return f"{len(self.png_files)} images"
    
    def view(self, index):
        image, mask = self.__getitem__(index)
        LunarCraterDataset.view_image(image, mask)

    @staticmethod
    def view_image(image, masks=None):
        denormalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(MEAN, STD)],
            std=[1/s for s in STD]
        )
        image = denormalize(image).detach().cpu()
        plt.imshow(torch.permute(image, (1, 2, 0)))

        if masks is not None:
            masks = masks.detach().cpu()
            n = masks.shape[0]
            colors = plt.get_cmap('hsv', n)
            for i in range(n):
                mask = masks[i].numpy()
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
                colored_mask[:, :, :3] = colors(i)[:3]  # RGB
                colored_mask[:, :, 3] = mask * 0.4      # Alpha
                plt.imshow(colored_mask, interpolation='none')

        plt.show()

    @staticmethod
    def view_bbox(image, target, class_name="Crater"):
        denormalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(MEAN, STD)], 
            std=[1/s for s in STD]
        )
        img_array = denormalize(image).cpu().numpy().transpose(1, 2, 0)
        img_array = np.clip(img_array, 0, 1)
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        H, W, C = img_array.shape

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_array)

        for i, (box, label) in enumerate(zip(boxes, labels)):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f'{class_name} ({i})', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))

        ax.set_title(f"Image Visualization ({len(boxes)} detections)")
        plt.show()

    @staticmethod
    def get_masks_from_json(json_name, w, h, n_att):
        with open(json_name) as f:
            data_dict = json.load(f)
        base_name = os.path.basename(json_name).split("_annotations.json")[0] + ".png"
        polygon_points = [e["points"] for e in data_dict.get(base_name, [])]
        num_masks = len(polygon_points)

        masks = torch.zeros((max(num_masks, n_att), w, h), dtype=torch.float)

        for i, points in enumerate(polygon_points):
            polygon_np = np.array(points)
            r = polygon_np[:, 1]
            c = polygon_np[:, 0]
            rr, cc = polygon(r, c, shape=(w, h))
            masks[i, rr, cc] = 1

        # Padding if num_masks < n_att
        return masks, num_masks

class TitaniumLunarDetectionDataset(Dataset):
    def __init__(self, json_path, image_dir, source_h, source_w):
        """
        Initializes the dataset by reading the COCO JSON annotations.
        """
        self.image_dir = image_dir
        # Standard ImageNet normalization values (often used when using ResNet backbones)

        # --- Transformations ---
        # We use standard PyTorch transforms
        self.source_h = source_h
        self.source_w = source_w

        transform = transforms.Compose([
            transforms.Resize((source_h, source_w)), # Resize all images to a fixed size
            transforms.ToTensor(), 
            transforms.Normalize(mean=MEAN, std=STD) 
        ])
        self.transform = transform
        
        # Load COCO JSON file
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        self.images_info = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']
        self.category_id = 1 # Crater category ID [4]

        # Map image IDs to their annotations
        self.img_to_ann = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_ann:
                self.img_to_ann[img_id] = []
            self.img_to_ann[img_id].append(ann)

        self.image_ids = list(self.images_info.keys())

    def __len__(self):
        """
        Returns the total number of images.
        """
        return len(self.image_ids)
    
    def __str__(self):
        return f"{len(self.image_ids)} images"

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]
        file_name = img_info['file_name']
        
        # 1. Load Image
        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        
        # --- A. CAPTURE ORIGINAL DIMENSIONS ---
        original_w, original_h = image.size # Width, Height
        
        # Determine Target Size (assuming uniform resize defined in your transform)
        # We assume your transform starts with T.Resize((TARGET_SIZE, TARGET_SIZE))
        # TARGET_SIZE = 600 # Assumes T.Resize is the first transform

        # Calculate Scaling Factors
        scale_y = self.source_h / original_h
        scale_x = self.source_w / original_w
        # ------------------------------------

        # 2. Extract and Convert Annotations
        annotations = self.img_to_ann.get(img_id, [])
        
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO format: [x_top_left, y_top_left, width, height]
            x_orig, y_orig, w_orig, h_orig = ann['bbox'] 
            
            # Skip tiny or invalid boxes
            if w_orig <= 0 or h_orig <= 0:
                continue

            # Convert to [x1, y1, x2, y2] format (original absolute coordinates)
            x1_orig = x_orig
            y1_orig = y_orig
            x2_orig = x_orig + w_orig
            y2_orig = y_orig + h_orig
            
            # --- B. APPLY SCALING TO BOX COORDINATES ---
            
            x1_scaled = x1_orig * scale_x
            y1_scaled = y1_orig * scale_y
            x2_scaled = x2_orig * scale_x
            y2_scaled = y2_orig * scale_y

            # Ensure coordinates are integers or properly float-tensed
            boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
            labels.append(self.category_id) 
        
        # 3. Assemble RetinaNet Targets List[Dict] format
        if boxes:
            # Convert to float32 tensor as expected by RetinaNet input (retinanet_targets)
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) 
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Handle empty image case
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }

        # 4. Apply Image Transformations (Image resized/normalized, but boxes are already scaled)
        if self.transform:
            image = self.transform(image)
            
        return image, target

def get_datasets(dataset_path, n_att, dim = None): # dataset_path="./fpsnet_dataset", n_att=N_ATT
    train_dataset = LunarCraterDataset(n_att=n_att, source_h=1200, source_w=1200, dataset_path=os.path.join(dataset_path, "train"))
    test_dataset = LunarCraterDataset(n_att=n_att, source_h=1200, source_w=1200, dataset_path=os.path.join(dataset_path, "test"))
    if dim is not None:
        train_dataset.resize(dim[0], dim[1])
        test_dataset.resize(dim[0], dim[1])
    return train_dataset, test_dataset

def titanium_collate_fn(batch):
    # 'batch' is a list of tuples: [(image1, target1), (image2, target2), ...]
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Stack images into a single tensor (B, C, H, W)
    images = torch.stack(images, 0)
    
    # Targets remain a list of dictionaries (List[Dict])
    return images, targets

def get_titanium_datasets(dataset_path, dim):
    train_dataset = TitaniumLunarDetectionDataset(
        json_path=os.path.join(dataset_path, "train", "_annotations.coco.json"),
        image_dir=os.path.join(dataset_path, "train"),
        source_h=dim[0],
        source_w=dim[1]
    )
    test_dataset = TitaniumLunarDetectionDataset(
        json_path=os.path.join(dataset_path, "valid", "_annotations.coco.json"),
        image_dir=os.path.join(dataset_path, "valid"),
        source_h=dim[0],
        source_w=dim[1]
    )
    return train_dataset, test_dataset