import torch
from src.models.retinanet import RetinaNet
from src.utils import remove_duplicates, inflate_annots, visualize_detections, remove_low_confidence_edge_boxes, prediction_boxes_to_dict
from src.io import load_model_weights
import os
from PIL import Image
from torchvision import transforms
import json
import sys
from src.io import save_json

MEAN=(0.485, 0.456, 0.406)
STD=(0.229, 0.224, 0.225)
image_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])

def annotate(image_batch: torch.Tensor):
    with torch.inference_mode():
        predictions = model(image_batch)
        final_predictions = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            image = image_batch[i]
            output_mask = prediction["scores"] >= SCORE_THRESH
            prediction = {
                "boxes": prediction["boxes"][output_mask].to(device),
                "labels": torch.tensor(list(range(len(output_mask)))).to(device),
                "scores": prediction["scores"][output_mask].to(device)
            }
            removed_duplicates = remove_duplicates(prediction, iou_threshold=-0.1)
            inflated_annots = inflate_annots(removed_duplicates, image.shape[1], image.shape[2], 0, 0.3)
            edge_cleaned_annots = remove_low_confidence_edge_boxes(inflated_annots, image.shape[1], image.shape[2], 0.5, 0.01)
            edge_cleaned_annots["labels"] = torch.tensor(list(range(len(edge_cleaned_annots["boxes"])))).to(device) # final labels
            final_predictions.append(edge_cleaned_annots)
        return final_predictions

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 annotate.py <inputs_path> <outputs_path> [<renderings_path>]")
        sys.exit(1)
    
    inputs_path = sys.argv[1]
    outputs_path = sys.argv[2]
    renderings_path = None
    if len(sys.argv) >= 4:
        renderings_path = sys.argv[3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RetinaNet(state_dict=load_model_weights("saved_weights", "annotator_weights.pth")).to(device)
    model.eval()

    inputs = os.listdir(inputs_path)
    path_inputs = [os.path.join(inputs_path, input) for input in inputs]
    file_names  = [ ".".join(i.split(".")[:-1]) for i in inputs ]
    json_file_names = [ f + ".json" for f in file_names ]

    os.makedirs(outputs_path, exist_ok=True)
    
    if renderings_path is not None:
        os.makedirs(renderings_path, exist_ok=True)
        rendered_file_names = [ f + ".png" for f in file_names ]
    
    SCORE_THRESH = 0.2
    BATCH_SIZE = 70

    no_of_batches = len(path_inputs) // BATCH_SIZE + (1 if len(path_inputs) % BATCH_SIZE != 0 else 0)

    for i in range(0, len(path_inputs), BATCH_SIZE):
        batch_inputs = path_inputs[i:min(i+BATCH_SIZE, len(path_inputs))]
        image_tensors = []

        for input_path in batch_inputs:
            image_tensors.append(image_transforms(Image.open(input_path)).to(device))
        
        image_tensors = torch.stack(image_tensors)
        
        print(f"Processing images: ({i+1}-{min(i+BATCH_SIZE, len(path_inputs))})/{len(path_inputs)}")
        
        predictions = annotate(image_tensors)
        
        image_tensors = image_tensors.cpu()
        
        for j, prediction in enumerate(predictions):
            image_tensor = image_tensors[j]
            json_dict = prediction_boxes_to_dict(prediction["boxes"])
            with open(os.path.join(outputs_path, json_file_names[i+j]), "w") as f:
                json.dump(json_dict, f, indent=2)

            if renderings_path is not None:
                visualize_detections(
                    image_tensor=image_tensor,
                    targets_dict=prediction,
                    mean=MEAN,
                    std=STD,
                    show_labels=True,
                    save_to_file=os.path.join(renderings_path, rendered_file_names[i+j])
                    )
        
        del image_tensors
        torch.cuda.empty_cache()