
import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mnist_cnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
}


# --------------------------------------------------
# 2. Image transformation
# Must match your training preprocessing
# --------------------------------------------------

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# --------------------------------------------------
# 3. CNN architecture
# Must match the architecture used in Colab
# --------------------------------------------------

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)


# --------------------------------------------------
# 4. FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Handwritten Digit Recognition",
    description="Recognize handwritten digits from 0 to 9.",
    version="2.0.0",
)

# Configure CORS so frontend can communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --------------------------------------------------
# 5. Serve frontend without a static folder
# --------------------------------------------------

@app.get("/", include_in_schema=False)
async def home():
    path = BASE_DIR / "index.html"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="index.html was not found beside main.py.",
        )

    return FileResponse(path)


@app.get("/static/style.css", include_in_schema=False)
async def stylesheet():
    path = BASE_DIR / "style.css"

    if not path.is_file():
        raise HTTPException(status_code=404, detail="style.css not found.")

    return FileResponse(path, media_type="text/css")


@app.get("/static/script.js", include_in_schema=False)
async def javascript():
    path = BASE_DIR / "script.js"

    if not path.is_file():
        raise HTTPException(status_code=404, detail="script.js not found.")

    return FileResponse(path, media_type="application/javascript")


# --------------------------------------------------
# 6. Load trained model
# --------------------------------------------------

model = None
model_load_error = None


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    network = MNISTCNN().to(DEVICE)

    try:
        state = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        # Support either a raw state_dict or a checkpoint
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        network.load_state_dict(state)
        network.eval()
        return network

    except Exception as exc:
        raise RuntimeError(
            "Could not load model weights. Check that mnist_cnn.pth "
            "was saved from the same MNISTCNN architecture."
        ) from exc


try:
    model = load_model()
    print("✓ Model loaded successfully.")
except Exception as e:
    model_load_error = str(e)
    print(f"✗ Failed to load model: {e}")


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if model loaded correctly
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable. Error: {model_load_error}",
        )

    # Validate file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # Read image
    try:
        content = await file.read()

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large.")

        image = Image.open(io.BytesIO(content))

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Not a valid image.")
    except Exception:
        raise HTTPException(status_code=500, detail="Could not read image.")

    # Preprocessing to match MNIST training:
    # 1. Convert to grayscale (L)
    # 2. Resize to 28x28
    # 3. Invert colors (MNIST is white on black, PIL opens as black on white)
    try:
        image = ImageOps.grayscale(image)
        image = image.resize((28, 28))
        image = ImageOps.invert(image)

        # Transform to tensor and normalize
        tensor = inference_transform(image)
        tensor = tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension

    except Exception:
        raise HTTPException(status_code=500, detail="Error during preprocessing.")

    # Inference
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
