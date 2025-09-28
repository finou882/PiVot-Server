"""
Multimodal Text Generation Demo
画像+テキスト → テキスト出力のデモ
"""
import requests
import json
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any

def create_demo_image() -> bytes:
    """デモ用の画像を生成"""
    # 簡単な色付きの画像を作成
    image = Image.new('RGB', (224, 224), color='lightblue')
    
    # 中央に四角形を描画
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 174, 174], fill='red', outline='black', width=2)
    draw.text((80, 100), "DEMO", fill='white')
    
    # バイトに変換
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG')
    return img_buffer.getvalue()

class MultimodalDemo:
    """マルチモーダル推論デモクラス"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def check_server_status(self) -> Dict[str, Any]:
        """サーバーの状態を確認"""
        try:
            response = requests.get(f"{self.base_url}/")
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "server_unavailable"}
    
    def test_multimodal_inference(self, 
                                image_bytes: bytes,
                                text_prompt: str,
                                model_name: str = "demo_model") -> Dict[str, Any]:
        """マルチモーダル推論をテスト"""
        try:
            files = {'image': ('demo.jpg', image_bytes, 'image/jpeg')}
            data = {
                'model_name': model_name,
                'text': text_prompt,
                'maintain_aspect_ratio': True,
                'return_features': False
            }
            
            response = requests.post(
                f"{self.base_url}/infer/multimodal",
                files=files,
                data=data
            )
            
            return response.json()
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def demonstrate_capabilities(self):
        """機能をデモンストレーション"""
        print("🎯 マルチモーダルテキスト生成デモ")
        print("=" * 50)
        
        # 1. サーバー状態確認
        print("1. サーバー状態確認...")
        status = self.check_server_status()
        if "error" in status:
            print(f"❌ サーバー接続エラー: {status['error']}")
            print("💡 サーバーを起動してください: python main.py")
            return
        
        print(f"✅ サーバー稼働中")
        print(f"   NPU利用可能: {status.get('npu_available', 'Unknown')}")
        print(f"   デバイス: {status.get('device', 'Unknown')}")
        print()
        
        # 2. デモ画像生成
        print("2. デモ画像生成...")
        demo_image = create_demo_image()
        print("✅ 224x224のデモ画像を生成しました")
        print("   内容: 青背景に赤い四角形と「DEMO」テキスト")
        print()
        
        # 3. 様々なプロンプトでテスト
        test_prompts = [
            "この画像に何が写っていますか？",
            "この画像の色を説明してください。",
            "この画像を見て短い物語を作ってください。",
            "この画像の幾何学的な特徴を説明してください。",
            "この画像からインスピレーションを得た詩を作ってください。"
        ]
        
        print("3. マルチモーダル推論テスト...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- テスト {i} ---")
            print(f"プロンプト: {prompt}")
            
            result = self.test_multimodal_inference(demo_image, prompt)
            
            if result.get('success'):
                print("✅ 推論成功")
                print(f"実行時間: {result['timing']['total_time']:.3f}秒")
                print(f"デバイス: {result.get('device', 'Unknown')}")
                
                # 生成されたテキストを表示
                if 'generated_text' in result:
                    print(f"生成テキスト: {result['generated_text']}")
                else:
                    print("Raw出力キー:", list(result.get('outputs', {}).keys()))
                    
            elif "Model not loaded" in str(result.get('error', '')):
                print("⚠️  モデルが読み込まれていません")
                print("💡 実際のモデル（CLIP, BLIP等）を読み込んでください")
            else:
                print(f"❌ 推論失敗: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 50)
        print("📝 実用化のための次のステップ:")
        print("1. マルチモーダルモデル（CLIP、BLIP、Flamingo等）をOpenVINO形式で用意")
        print("2. /models/loadエンドポイントでモデルを読み込み")
        print("3. 実際の画像ファイルで推論を実行")
        print("4. 生成されたテキストを用途に応じて後処理")

def main():
    """メイン実行"""
    demo = MultimodalDemo()
    demo.demonstrate_capabilities()
    
    print("\n🔧 開発者向けコード例:")
    print("""
# Python クライアントコード例
import requests

# 画像+テキストでテキスト生成
with open('your_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {
        'model_name': 'blip_model',  # 実際のモデル名
        'text': '这张图片描述了什么？',
        'maintain_aspect_ratio': True
    }
    response = requests.post('http://localhost:8000/infer/multimodal', 
                           files=files, data=data)
    
result = response.json()
if result['success'] and 'generated_text' in result:
    print("生成された回答:", result['generated_text'])
""")

if __name__ == "__main__":
    main()