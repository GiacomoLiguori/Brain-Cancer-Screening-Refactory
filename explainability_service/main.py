import io
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from contextlib import asynccontextmanager

from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor
)

from fastapi import (
    FastAPI, 
    UploadFile, 
    File
)
from fastapi.responses import StreamingResponse

MODEL_PATH = "./resnet"

model = None
layer = None
image_processor = None

@asynccontextmanager
async def lifespan(app : FastAPI):
    global model, layer, image_processor, features, handle_feat, handle_grad

    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)

    layer = model.resnet.encoder.stages[-1].layers[-1]

    model.eval()
    yield

    image_processor = None
    model = None
    layer = None

app = FastAPI(title = "Brain Cancer Classification", description = "Microservizio per generare la mappa di calore", lifespan = lifespan)

@app.post("/explain")
async def predict(file : UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    inputs = image_processor(image, return_tensors = 'pt')
    px_values = inputs['pixel_values']
    px_values.requires_grad = True

    features = dict()

    def get_features_hook(module, input, output):
        features['data'] = output

    def get_gradients_hook(module, grad_input, grad_output):
        features['grad'] = grad_output[0]

    handle_feat = layer.register_forward_hook(get_features_hook)
    handle_grad = layer.register_full_backward_hook(get_gradients_hook)

    try:
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim = -1).item()

        model.zero_grad()

        logits[0, predicted].backward()

        handle_feat.remove()
        handle_grad.remove()

        gradients = features['grad']
        activations = features['data']

        ch_weights = torch.mean(gradients, dim = (2, 3), keepdim = True)

        grad_cam = torch.sum(ch_weights * activations, dim = 1).squeeze()
        grad_cam = torch.nn.functional.relu(grad_cam)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)

        grad_cam_np = grad_cam.detach().cpu().numpy()

        w, h = image.size
        resized_grad_cam = Image.fromarray((grad_cam_np * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)

        cmap = plt.get_cmap('jet')
        grad_cam_cl = np.array(cmap(np.array(resized_grad_cam) / 255.0))[:, :, :3]
        grad_cam_cl = Image.fromarray((grad_cam_cl * 255).astype(np.uint8))

        blended_image = Image.blend(image, grad_cam_cl, alpha = 0.5)

        buffer = io.BytesIO()
        blended_image.save(buffer, format = "PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type = "image/png")
    
    finally:
        handle_feat.remove()
        handle_grad.remove()