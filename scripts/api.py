from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.predict import predict_image

app = FastAPI(title="Pig Detector API")

CFG_PATH = "config.yaml"
WEIGHTS_PATH = "models/best_model.pth"

class PredictRequest(BaseModel):
    image_path: str

@app.post("/predict")
def predict(req: PredictRequest):
    results = predict_image(CFG_PATH, WEIGHTS_PATH, req.image_path)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

