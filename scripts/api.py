import os
import sys
import tempfile
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── 一：动态定位项目根目录 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── 二：导入自定义的模型加载与预测函数 ───────────────────────────────
from scripts.predict import load_model, predict_image

# ─── 三：配置文件和权重文件路径 ────────────────────────────────────
CFG_PATH = ROOT / "config.yaml"
WEIGHTS_PATH = ROOT / "models" / "best_model.pth"

# ─── 四：FastAPI 应用初始化 ─────────────────────────────────────────
app = FastAPI(title="Pig Detector API", version="1.0")

# ─── 五：请求/响应模型 ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_path: str
    conf: float = 0.5
    top_k: int = 10

class PredictResponse(BaseModel):
    results: list

# ─── 六：启动时加载模型 ─────────────────────────────────────────────
try:
    MODEL, MODEL_META = load_model(str(CFG_PATH), str(WEIGHTS_PATH))
    print(f"✅ 模型加载成功: {CFG_PATH}, {WEIGHTS_PATH}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 如果模型加载失败，通常应该退出程序
    sys.exit(1)

# ─── 七：预测接口 ───────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 支持远程 URL
    image_src = req.image_path
    if image_src.startswith(("http://", "https://")):
        try:
            resp = requests.get(image_src, stream=True)
            resp.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无法下载图片: {e}")
        suffix = Path(image_src).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.flush()
        image_path = tmp.name
    else:
        image_path = image_src

    # 本地文件检查
    if not Path(image_path).exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

    # 调用预测并返回
    try:
        results = predict_image(str(CFG_PATH), str(WEIGHTS_PATH), image_path,
                                conf=req.conf, top_k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")

    return PredictResponse(results=results)

# ─── 八：本地开发时启动 ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "scripts.api:app",
        host="0.0.0.0",
        port=8093,
        reload=True,
        log_level="info"
    )
