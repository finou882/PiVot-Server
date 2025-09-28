"""
Simplified Image ID Based Client
簡素化された画像IDベースクライアント
"""
import requests
import json
import time
from typing import Dict, Any, Optional

class SimpleImageIDClient:
    """簡素化された画像IDベースの推論クライアント"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def infer_text(self,
                   model_name: str,
                   text: str, 
                   image_id: str,
                   image_base_url: str = "http://pi.local/image") -> Dict[str, Any]:
        """
        画像IDとテキストからテキストを生成
        
        Args:
            model_name: 使用するモデル名
            text: テキストプロンプト
            image_id: 画像のID
            image_base_url: 画像サーバーのベースURL
        
        Returns:
            推論結果（generated_textを含む）
        """
        try:
            payload = {
                "model_name": model_name,
                "text": text,
                "image_id": image_id,
                "image_base_url": image_base_url,
                "maintain_aspect_ratio": True
            }
            
            response = requests.post(
                f"{self.base_url}/infer/text-from-image-id",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            return response.json()
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def infer_text_form(self,
                       model_name: str,
                       text: str,
                       image_id: str,
                       image_base_url: str = "http://pi.local/image") -> Dict[str, Any]:
        """
        画像IDとテキストからテキストを生成（Form版）
        """
        try:
            data = {
                "model_name": model_name,
                "text": text,
                "image_id": image_id,
                "image_base_url": image_base_url,
                "maintain_aspect_ratio": True
            }
            
            response = requests.post(
                f"{self.base_url}/infer/text-from-image-id/form",
                data=data
            )
            
            return response.json()
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def load_model(self, model_name: str, model_path: str, model_type: str = "multimodal") -> Dict[str, Any]:
        """モデルを読み込み"""
        try:
            data = {
                "model_name": model_name,
                "model_path": model_path,
                "model_type": model_type
            }
            
            response = requests.post(f"{self.base_url}/models/load", data=data)
            return response.json()
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_server(self) -> bool:
        """サーバーの稼働状態をチェック"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False

def main():
    """使用例"""
    print("🎯 PiVot-Server 簡素化クライアント")
    print("=" * 50)
    
    client = SimpleImageIDClient()
    
    # サーバー接続確認
    if not client.check_server():
        print("❌ サーバーに接続できません")
        print("💡 サーバーを起動してください: python main.py")
        return
    
    print("✅ サーバー接続OK")
    print()
    
    # 使用例
    print("📖 使用例:")
    print("""
# 基本的な推論
client = SimpleImageIDClient()

result = client.infer_text(
    model_name="multimodal_model",
    text="この画像に何が写っていますか？",
    image_id="sample_001.jpg", 
    image_base_url="http://pi.local/image"
)

if result['success']:
    print("生成テキスト:", result['generated_text'])
    print("推論時間:", result['timing']['total_time'])
else:
    print("エラー:", result['error'])
""")
    
    print("\n💡 実際に推論を実行するには:")
    print("1. マルチモーダルモデルを読み込み")
    print("2. 画像サーバーを起動")
    print("3. 実際の画像IDを指定")

if __name__ == "__main__":
    main()