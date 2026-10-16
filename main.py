
import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
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
except Exception as exc:
    model_load_error = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------
# 7. Convert uploaded image to MNIST-style input
# --------------------------------------------------

def prepare_handwriting(image: Image.Image) -> torch.Tensor:
    # Correct camera orientation and use grayscale
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.autocontrast(image)

    # Estimate background brightness from the four corners
    width, height = image.size

    corners = [
        image.getpixel((0, 0)),
        image.getpixel((width - 1, 0)),
        image.getpixel((0, height - 1)),
        image.getpixel((width - 1, height - 1)),
    ]

    background = sorted(corners)[len(corners) // 2]

    # MNIST convention: black background, light digit
    if background > 127:
        image = ImageOps.invert(image)

    # Remove faint background noise
    image = image.point(lambda pixel: 255 if pixel > 80 else 0)

    # Find the digit and place it in the center of a 28x28 canvas
    bbox = image.getbbox()
    canvas = Image.new("L", (28, 28), 0)

    if bbox:
        digit = image.crop(bbox)
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

        x_offset = (28 - digit.width) // 2
        y_offset = (28 - digit.height) // 2

        canvas.paste(digit, (x_offset, y_offset))

    # Convert to tensor and normalize as during MNIST training
    tensor = inference_transform(canvas)
    return tensor.unsqueeze(0).to(DEVICE)


# --------------------------------------------------
# 8. Health check
# --------------------------------------------------

@app.get("/api/health")
async def health():
    if model is None:
        return {
            "status": "model_missing",
            "detail": model_load_error,
        }

    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": MODEL_PATH.name,
    }


# --------------------------------------------------
# 9. Prediction API
# --------------------------------------------------

@app.post("/api/predict")
async def predict_digit(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded: {model_load_error}",
        )

    try:
        if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Upload a PNG, JPG, JPEG, WEBP, or BMP image.",
            )

        raw_bytes = await file.read()

        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Maximum image size is 8 MB.",
            )

        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        if image.width < 5 or image.height < 5:
            raise HTTPException(
                status_code=400,
                detail="Image dimensions are too small.",
            )

        if image.width > 6000 or image.height > 6000:
            raise HTTPException(
                status_code=413,
                detail="Image dimensions are too large.",
            )

        input_tensor = prepare_handwriting(image)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

            top_scores, top_indices = torch.topk(probabilities, k=3)

        predicted_digit = int(top_indices[0].item())
        confidence = float(top_scores[0].item()) * 100

        top_predictions = [
            {
                "digit": int(digit.item()),
                "confidence": round(float(score.item()) * 100, 2),
            }
            for score, digit in zip(top_scores, top_indices)
        ]

        return {
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2),
            "top_predictions": top_predictions,
            "filename": Path(file.filename or "image").name,
            "note": (
                "Confidence is the model's estimated probability, "
                "not a guarantee of correctness."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        await file.close()
