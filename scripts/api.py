from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.predict import load_model, predict_image_with_model

app = FastAPI(title="Pig Detector API")

CFG_PATH = "config.yaml"
WEIGHTS_PATH = "models/best_model.pth"

# Load the model once at startup
MODEL, MODEL_META = load_model(CFG_PATH, WEIGHTS_PATH)


class PredictRequest(BaseModel):
    image_path: str


@app.post("/predict")
def predict(req: PredictRequest):
    results = predict_image_with_model(MODEL, req.image_path)
    return {"results": results}


@app.get("/version")
def version():
    """Return model version and training time if available."""
    return {
        "version": MODEL_META.get("version", "unknown"),
        "trained_at": MODEL_META.get("trained_at", "unknown"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

