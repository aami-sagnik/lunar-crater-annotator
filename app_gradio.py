import torch
import gradio as gr
import tempfile
import os
import json
from PIL import Image
from torchvision import transforms

from src.models.retinanet import RetinaNet
from src.utils import (
    remove_duplicates,
    inflate_annots,
    visualize_detections,
    remove_low_confidence_edge_boxes,
    prediction_boxes_to_dict,
)
from src.io import load_model_weights

# -----------------------------------
# CONFIG
# -----------------------------------
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SCORE_THRESH = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

image_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# -----------------------------------
# LOAD MODEL (once)
# -----------------------------------
model = RetinaNet(
    state_dict=load_model_weights("saved_weights", "annotator_weights.pth")
).to(device)
model.eval()

# -----------------------------------
# ANNOTATION FUNCTION
# -----------------------------------
def annotate(image: Image.Image):
    image_tensor = image_transforms(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.inference_mode():
        predictions = model(image_tensor)

        prediction = predictions[0]
        image_t = image_tensor[0]

        output_mask = prediction["scores"] >= SCORE_THRESH
        prediction = {
            "boxes": prediction["boxes"][output_mask].to(device),
            "labels": torch.tensor(list(range(len(output_mask)))).to(device),
            "scores": prediction["scores"][output_mask].to(device),
        }

        removed_duplicates = remove_duplicates(prediction, iou_threshold=-0.1)
        inflated_annots = inflate_annots(
            removed_duplicates,
            image_t.shape[1],
            image_t.shape[2],
            0,
            0.10,
        )

        edge_cleaned_annots = remove_low_confidence_edge_boxes(
            inflated_annots,
            image_t.shape[1],
            image_t.shape[2],
            0.5,
            0.01,
        )

        edge_cleaned_annots["labels"] = torch.tensor(
            list(range(len(edge_cleaned_annots["boxes"])))
        ).to(device)

    # Render annotated image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        visualize_detections(
            image_tensor=image_t.cpu(),
            targets_dict=edge_cleaned_annots,
            mean=MEAN,
            std=STD,
            show_labels=True,
            save_to_file=tmp_img.name,
        )
        rendered_image = Image.open(tmp_img.name)

    json_dict = prediction_boxes_to_dict(edge_cleaned_annots["boxes"])

    return rendered_image, json.dumps(json_dict, indent=2)


# -----------------------------------
# STABLE DIFFUSION STYLE UI
# -----------------------------------
with gr.Blocks(css="""
    .gradio-container {
        height: 100vh;
        overflow-y: auto;
    }
""") as demo:

    gr.Markdown("# 🖼️ Image Annotation")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image (1200x1200)",
                type="pil",
                height=600
            )

        with gr.Column():
            output_image = gr.Image(
                label="Annotated Output",
                height=600
            )

    json_output = gr.Textbox(
        label="JSON Output",
        lines=8
    )

    # Auto run when image changes (no button)
    input_image.change(
        fn=annotate,
        inputs=input_image,
        outputs=[output_image, json_output],
    )

demo.launch()