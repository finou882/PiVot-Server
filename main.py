"""
PiVot-Server: Lightweight Image ID Based Multimodal Inference Server
画像IDベースのマルチモーダル推論サーバー（簡素化版）
"""
import os
import time
import logging
import uvicorn
import aiohttp
import asyncio
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json

# 独自モジュールのインポート
from npu_manager import npu_manager
from npu_inference_engine import npu_inference_engine

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 画像取得ユーティリティ
async def fetch_image_from_url(image_url: str, timeout: int = 10) -> Optional[bytes]:
    """URLから画像をダウンロード"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'image' in content_type:
                        image_bytes = await response.read()
                        logger.info(f"Image downloaded: {len(image_bytes)} bytes from {image_url}")
                        return image_bytes
                    else:
                        logger.error(f"Invalid content type: {content_type} for {image_url}")
                        return None
                else:
                    logger.error(f"HTTP {response.status} when fetching {image_url}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout when fetching image from {image_url}")
        return None
    except Exception as e:
        logger.error(f"Error fetching image from {image_url}: {e}")
        return None

def build_image_url(base_url: str, image_id: str) -> str:
    """画像URLを構築"""
    if base_url.endswith('/'):
        return f"{base_url}{image_id}"
    else:
        return f"{base_url}/{image_id}"

# FastAPIアプリ初期化
app = FastAPI(
    title="PiVot-Server",
    description="Lightweight Image ID Based Multimodal Inference Server",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydanticモデル定義
class TextOnlyInferenceRequest(BaseModel):
    model_name: str
    text: str
    image_id: str
    image_base_url: Optional[str] = "http://pi.local/image"
    maintain_aspect_ratio: Optional[bool] = True
    config: Optional[Dict[str, Any]] = None

class TextOnlyInferenceResponse(BaseModel):
    success: bool
    generated_text: Optional[str] = None
    model_name: str
    timing: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    device: Optional[str] = None
    image_url: Optional[str] = None

# ヘルスチェックエンドポイント
@app.get("/")
async def root():
    """ルートエンドポイント - システム情報を返す"""
    return {
        "message": "PiVot-Server (Image ID Based) is running",
        "version": "1.0.0",
        "npu_available": npu_manager.is_npu_available(),
        "device": npu_manager.npu_device,
        "endpoints": [
            "POST /infer/text-from-image-id",
            "POST /infer/text-from-image-id/form",
            "POST /models/load"
        ]
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    import datetime
    return {"status": "healthy", "timestamp": str(datetime.datetime.now())}

# モデル管理エンドポイント（簡素版）
@app.post("/models/load")
async def load_model(
    model_name: str = Form(...),
    model_path: str = Form(...),
    model_type: str = Form(default="multimodal")
):
    """モデルを読み込み"""
    try:
        success = npu_inference_engine.load_model(
            model_name=model_name,
            model_path=model_path,
            model_type=model_type
        )
        
        if success:
            return {"success": True, "message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")
            
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# マルチモーダル推論エンドポイント（画像ID使用）
@app.post("/infer/text-from-image-id", response_model=TextOnlyInferenceResponse)
async def infer_text_from_image_id(request: TextOnlyInferenceRequest):
    """
    画像IDとテキストからテキストを生成
    
    Args:
        request.model_name: 使用するモデル名
        request.text: テキストプロンプト
        request.image_id: 画像のID（例: "abc123.jpg"）
        request.image_base_url: 画像サーバーのベースURL（デフォルト: "http://pi.local/image"）
        request.maintain_aspect_ratio: アスペクト比を保持するか
        request.config: 追加の前処理設定
    
    Returns:
        TextOnlyInferenceResponse: 生成されたテキストのみを含む応答
    """
    try:
        start_time = time.time()
        
        # 画像URLを構築
        image_url = build_image_url(request.image_base_url, request.image_id)
        logger.info(f"Fetching image from: {image_url}")
        
        # 画像をダウンロード
        image_bytes = await fetch_image_from_url(image_url)
        if image_bytes is None:
            return TextOnlyInferenceResponse(
                success=False,
                model_name=request.model_name,
                error=f"Failed to fetch image from {image_url}",
                image_url=image_url
            )
        
        fetch_time = time.time() - start_time
        
        # 前処理設定の準備
        preprocess_config = request.config or {}
        preprocess_config['maintain_aspect_ratio'] = request.maintain_aspect_ratio
        
        # マルチモーダル推論実行
        result = npu_inference_engine.infer_multimodal(
            model_name=request.model_name,
            image_input=image_bytes,
            text_input=request.text,
            preprocess_config=preprocess_config
        )
        
        if result is None:
            return TextOnlyInferenceResponse(
                success=False,
                model_name=request.model_name,
                error="Multimodal inference failed",
                image_url=image_url
            )
        
        # 成功時のレスポンス
        if result.get('success'):
            # タイミング情報にフェッチ時間を追加
            timing = result.get('timing', {})
            timing['image_fetch_time'] = fetch_time
            
            return TextOnlyInferenceResponse(
                success=True,
                generated_text=result.get('generated_text', 'テキスト生成に失敗しました'),
                model_name=request.model_name,
                timing=timing,
                device=result.get('device'),
                image_url=image_url
            )
        else:
            return TextOnlyInferenceResponse(
                success=False,
                model_name=request.model_name,
                error=result.get('error', 'Unknown inference error'),
                image_url=image_url
            )
            
    except Exception as e:
        logger.error(f"Text-from-image-id inference failed: {e}")
        return TextOnlyInferenceResponse(
            success=False,
            model_name=request.model_name,
            error=str(e)
        )

# Form版のマルチモーダル推論エンドポイント（画像ID使用）
@app.post("/infer/text-from-image-id/form", response_model=TextOnlyInferenceResponse)
async def infer_text_from_image_id_form(
    model_name: str = Form(...),
    text: str = Form(...),
    image_id: str = Form(...),
    image_base_url: str = Form(default="http://pi.local/image"),
    maintain_aspect_ratio: bool = Form(default=True),
    config: Optional[str] = Form(default=None)
):
    """
    FormデータでIDから画像を取得してテキスト生成（ブラウザ等からの利用に便利）
    """
    try:
        # configをJSONからパース
        parsed_config = None
        if config:
            try:
                parsed_config = json.loads(config)
            except json.JSONDecodeError:
                logger.warning("Invalid config JSON, using defaults")
        
        # Pydanticモデルに変換
        request = TextOnlyInferenceRequest(
            model_name=model_name,
            text=text,
            image_id=image_id,
            image_base_url=image_base_url,
            maintain_aspect_ratio=maintain_aspect_ratio,
            config=parsed_config
        )
        
        # 本体の処理を呼び出し
        return await infer_text_from_image_id(request)
        
    except Exception as e:
        logger.error(f"Form-based text-from-image-id inference failed: {e}")
        return TextOnlyInferenceResponse(
            success=False,
            model_name=model_name,
            error=str(e)
        )

if __name__ == "__main__":
    # サーバー起動設定
    import argparse
    
    parser = argparse.ArgumentParser(description="PiVot-Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info("Starting PiVot-Server (Image ID Based)...")
    logger.info(f"NPU Available: {npu_manager.is_npu_available()}")
    logger.info(f"Device: {npu_manager.npu_device}")
    logger.info("Available endpoints:")
    logger.info("  POST /infer/text-from-image-id")
    logger.info("  POST /infer/text-from-image-id/form") 
    logger.info("  POST /models/load")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )