# scripts/api.py
import tempfile
import sys
from pathlib import Path
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 动态定位项目根目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.predict import load_model, predict_image

app = FastAPI(title="Pig Detector API", version="1.0")

# 配置文件和权重文件路径，使用绝对路径
CFG_PATH = ROOT / "config.yaml"
WEIGHTS_PATH = ROOT / "models" / "best_model.pth"

class PredictRequest(BaseModel):
    image_path: str

class PredictResponse(BaseModel):
    results: list

# 启动时加载模型，避免每次请求重复初始化
try:
    MODEL, MODEL_META = load_model(str(CFG_PATH), str(WEIGHTS_PATH))
    print(f"✅ 模型加载成功: {CFG_PATH}, {WEIGHTS_PATH}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    image_src = req.image_path
    # 支持远程 URL
    if image_src.startswith(("http://", "https://")):
        try:
            resp = requests.get(image_src, stream=True)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法下载图片: {e}")
        suffix = Path(image_src).suffix or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        for chunk in resp.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.flush()
        image_path = temp_file.name
    else:
        image_path = image_src

    # 本地文件检查
    if not Path(image_path).exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

    # 调用预测并返回
    try:
        results = predict_image(str(CFG_PATH), str(WEIGHTS_PATH), image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")

    return PredictResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "scripts.api:app",
        host="0.0.0.0",
        port=8093,
        reload=True,
        log_level="info"
    )
