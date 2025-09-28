"""
Mock Image Server for Testing
テスト用の画像サーバー（pi.localの代用）
"""
import os
import io
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock画像サーバーアプリ
mock_app = FastAPI(title="Mock Image Server", version="1.0.0")

# サンプル画像の生成関数
def create_sample_image(image_id: str, size: tuple = (224, 224)):
    """サンプル画像を動的生成"""
    from PIL import ImageDraw, ImageFont
    
    # 背景色をIDに基づいて決定
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    bg_color = colors[hash(image_id) % len(colors)]
    
    # 画像作成
    image = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # 中央に四角形を描画
    margin = size[0] // 8
    draw.rectangle([margin, margin, size[0]-margin, size[1]-margin], 
                  fill='white', outline='black', width=2)
    
    # テキストを描画
    try:
        # システムフォントを試行
        font = ImageFont.load_default()
    except:
        font = None
    
    # 画像IDを描画
    text_lines = [
        f"ID: {image_id[:12]}",
        f"Size: {size[0]}x{size[1]}",
        "Mock Image"
    ]
    
    text_y = size[1] // 2 - 30
    for line in text_lines:
        if font:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
        else:
            text_width = len(line) * 6  # 概算
        
        text_x = (size[0] - text_width) // 2
        draw.text((text_x, text_y), line, fill='black', font=font)
        text_y += 20
    
    return image

@mock_app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Mock Image Server for PiVot-Server Testing",
        "version": "1.0.0",
        "endpoints": {
            "get_image": "/image/{image_id}",
            "list_samples": "/samples"
        }
    }

@mock_app.get("/image/{image_id}")
async def get_image(image_id: str):
    """
    画像IDから画像を取得
    
    Args:
        image_id: 画像のID（例: "sample_001.jpg", "test.png"）
    """
    try:
        logger.info(f"Image requested: {image_id}")
        
        # 画像生成
        image = create_sample_image(image_id)
        
        # バイトに変換
        img_buffer = io.BytesIO()
        
        # ファイル拡張子に基づいてフォーマット決定
        if image_id.lower().endswith('.png'):
            format_type = 'PNG'
            content_type = 'image/png'
        elif image_id.lower().endswith(('.jpg', '.jpeg')):
            format_type = 'JPEG'
            content_type = 'image/jpeg'
        else:
            format_type = 'JPEG'  # デフォルト
            content_type = 'image/jpeg'
        
        image.save(img_buffer, format=format_type)
        img_buffer.seek(0)
        
        logger.info(f"Image served: {image_id} ({len(img_buffer.getvalue())} bytes)")
        
        return Response(
            content=img_buffer.getvalue(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={image_id}",
                "Cache-Control": "public, max-age=3600"
            }
        )
        
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

@mock_app.get("/samples")
async def list_samples():
    """利用可能なサンプル画像IDのリスト"""
    sample_ids = [
        "sample_001.jpg",
        "sample_002.jpg", 
        "test_image.png",
        "demo_photo.jpg",
        "artwork_001.jpg",
        "landscape_001.jpg",
        "portrait_001.jpg",
        "product_001.jpg",
        "scene_001.jpg",
        "upload_12345.jpg",
        "upload_67890.jpg"
    ]
    
    return {
        "available_samples": sample_ids,
        "note": "これらのIDで /image/{image_id} にアクセスできます",
        "example_urls": [
            f"http://localhost:8001/image/{sample_id}"
            for sample_id in sample_ids[:3]
        ]
    }

@mock_app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "service": "mock_image_server"}

def start_mock_server():
    """モックサーバーを起動"""
    print("🖼️  Mock Image Server Starting...")
    print("   URL: http://localhost:8001")
    print("   Sample images: http://localhost:8001/samples")
    print("   Image endpoint: http://localhost:8001/image/{image_id}")
    print()
    
    uvicorn.run(
        "mock_image_server:mock_app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    start_mock_server()