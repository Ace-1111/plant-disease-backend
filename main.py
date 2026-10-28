import io
import gc
import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Single thread force karein taaki PyTorch CPU par multiple threads se RAM spike na kare
torch.set_num_threads(1)

app = FastAPI(title="Plant Disease Detection API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Robust Path Handling -----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_mobilenetv2_field_ready.pth"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
REMEDIES_PATH = BASE_DIR / "data" / "remedies.json"

DEVICE = torch.device("cpu")

# Load Class Names
if not CLASSES_PATH.exists():
    raise FileNotFoundError(f"Class names file not found at: {CLASSES_PATH}")

with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

# Load Remedies Database
remedies_db = {}
if REMEDIES_PATH.exists():
    try:
        with open(REMEDIES_PATH, "r", encoding="utf-8") as f:
            remedies_db = json.load(f)
    except Exception as e:
        print(f"Warning: Could not parse remedies.json: {e}")

# Balanced transforms (Optimized for real farm patterns)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- Load Model -----------------
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    net = models.mobilenet_v2(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_features, len(class_names))
    
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.to(DEVICE)
    net.eval()
    
    # Unused weights aur memory release
    del state_dict
    gc.collect()
    return net

model = get_model()

# RAM-friendly Nature Shade Filter (Thumbnail sample to save ~120MB RAM)
def inspect_nature_shades(image: Image.Image) -> bool:
    # 64x64 thumbnail par check karo, poori full-res image ko memory me list banane ki zaroorat nahi
    small_img = image.copy()
    small_img.thumbnail((64, 64))
    hsv_image = small_img.convert("HSV")
    h_channel = hsv_image.split()[0]
    
    h_data = h_channel.getdata()
    matching_pixels = sum(1 for h in h_data if (30 <= h <= 85) or (10 <= h < 30) or (85 < h <= 110))
    total_pixels = len(h_data)
    
    del small_img, hsv_image, h_channel
    return (matching_pixels / total_pixels) > 0.08

# ----------------- Endpoints -----------------
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Plant Disease Classifier API is ready",
        "total_classes": len(class_names)
    }

@app.post("/predict")
async def predict_leaf(file: UploadFile = File(...)):
    filename = file.filename.lower() if file.filename else ""
    valid_extensions = (".jpg", ".jpeg", ".png")
    valid_content_types = ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]

    if not (filename.endswith(valid_extensions) or file.content_type in valid_content_types):
        raise HTTPException(
            status_code=400,
            detail="Sirf JPEG ya PNG image upload karein."
        )

    image = None
    tensor = None
    try:
        # Read and convert image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        del image_bytes
        
        # OOD Natural Filter
        if not inspect_nature_shades(image):
            return {
                "success": False,
                "is_leaf": False,
                "message": "Koi plant leaf ya natural foliage detect nahi hui. Kripya kisi paudhe ke patte ki saaf photo upload karein.",
                "confidence": "N/A"
            }

        # Preprocessing & Inference
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = class_names[predicted_idx.item()]
        confidence_score = round(confidence.item() * 100, 2)

        # 1. Soft Out-of-Distribution Rejection Handling
        if predicted_class == "Background_Without_Leaves":
            if confidence_score > 60.0:
                return {
                    "success": False,
                    "is_leaf": False,
                    "message": "Koi plant leaf detect nahi hui. Kripya kisi paudhe ke patte ki saaf photo upload karein.",
                    "confidence": confidence_score
                }

        # 2. Field threshold lowered from 60.0% to 42.0%
        if confidence_score < 42.0:
            return {
                "success": False,
                "is_leaf": True,
                "message": "Patte ki bimari pehchan me nahi aa rahi. Kripya thoda paas se ya saaf roshni me photo lein.",
                "confidence": confidence_score
            }

        # 3. Label Formatting
        parts = predicted_class.split("___")
        crop_name = parts[0].replace("_", " ")
        disease_name = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
        
        if crop_name.lower() == "background without leaves":
            return {
                "success": False,
                "is_leaf": False,
                "message": "Koi plant leaf detect nahi hui. Kripya kisi paudhe ke patte ki photo upload karein.",
                "confidence": confidence_score
            }

        is_healthy = "healthy" in disease_name.lower()

        # Remedies fetch
        remedy = remedies_db.get(predicted_class, {
            "organic": "Prabhavit patto ko alag karein aur organic neem oil spray karein.",
            "chemical": "Uchit fungicide ka chhidkaw karein.",
            "prevention": "Paudho me hawa ka circulation banaye rakhein aur mitti ko clean rakhein."
        })

        return {
            "success": True,
            "is_leaf": True,
            "crop": crop_name,
            "disease": disease_name,
            "is_healthy": is_healthy,
            "confidence": f"{confidence_score}%",
            "treatment": remedy
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        # Har request ke baad memory purge karo
        del image, tensor
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)