"""
Image ID Based Multimodal Inference Client
画像IDを使用したマルチモーダル推論クライアント
"""
import requests
import json
import time
from typing import Dict, Any, Optional

class ImageIDInferenceClient:
    """画像IDベースの推論クライアント"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def infer_text_from_image_id(self,
                                model_name: str,
                                text: str, 
                                image_id: str,
                                image_base_url: str = "http://pi.local/image",
                                maintain_aspect_ratio: bool = True,
                                config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        画像IDとテキストからテキストを生成（JSON版）
        
        Args:
            model_name: 使用するモデル名
            text: テキストプロンプト
            image_id: 画像のID
            image_base_url: 画像サーバーのベースURL
            maintain_aspect_ratio: アスペクト比を保持するか
            config: 追加設定
        """
        try:
            payload = {
                "model_name": model_name,
                "text": text,
                "image_id": image_id,
                "image_base_url": image_base_url,
                "maintain_aspect_ratio": maintain_aspect_ratio,
                "config": config
            }
            
            response = requests.post(
                f"{self.base_url}/infer/text-from-image-id",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            return response.json()
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def infer_text_from_image_id_form(self,
                                     model_name: str,
                                     text: str,
                                     image_id: str,
                                     image_base_url: str = "http://pi.local/image",
                                     maintain_aspect_ratio: bool = True,
                                     config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        画像IDとテキストからテキストを生成（Form版）
        """
        try:
            data = {
                "model_name": model_name,
                "text": text,
                "image_id": image_id,
                "image_base_url": image_base_url,
                "maintain_aspect_ratio": maintain_aspect_ratio
            }
            
            if config:
                data["config"] = json.dumps(config)
            
            response = requests.post(
                f"{self.base_url}/infer/text-from-image-id/form",
                data=data
            )
            
            return response.json()
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# 使用例関数
def example_image_captioning():
    """画像キャプション生成の例"""
    print("=== 画像IDベース キャプション生成例 ===")
    
    client = ImageIDInferenceClient()
    
    # 画像IDを使用してキャプション生成
    result = client.infer_text_from_image_id(
        model_name="clip_caption_model",
        text="この画像に何が写っていますか？詳しく説明してください。",
        image_id="sample_001.jpg",
        image_base_url="http://pi.local/image"
    )
    
    if result.get('success'):
        print(f"✅ 推論成功")
        print(f"生成テキスト: {result['generated_text']}")
        print(f"推論時間: {result['timing']['total_time']:.3f}秒")
        print(f"画像取得時間: {result['timing']['image_fetch_time']:.3f}秒")
        print(f"使用デバイス: {result['device']}")
        print(f"画像URL: {result['image_url']}")
    else:
        print(f"❌ 推論失敗: {result.get('error')}")

def example_visual_qa():
    """視覚的質問応答の例"""
    print("\n=== 画像IDベース VQA例 ===")
    
    client = ImageIDInferenceClient()
    
    # 複数の質問で同じ画像を使用
    questions = [
        "この画像には何人の人が写っていますか？",
        "写っている人は何をしていますか？",
        "背景には何がありますか？",
        "この場面の時間帯はいつ頃だと思いますか？"
    ]
    
    image_id = "scene_001.jpg"
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- 質問 {i} ---")
        print(f"質問: {question}")
        
        result = client.infer_text_from_image_id(
            model_name="vqa_model",
            text=question,
            image_id=image_id,
            image_base_url="http://pi.local/image"
        )
        
        if result.get('success'):
            print(f"回答: {result['generated_text']}")
            print(f"推論時間: {result['timing']['total_time']:.3f}秒")
        else:
            print(f"エラー: {result.get('error')}")

def example_batch_processing():
    """バッチ処理の例"""
    print("\n=== バッチ処理例 ===")
    
    client = ImageIDInferenceClient()
    
    # 複数の画像IDとプロンプトのペア
    tasks = [
        ("photo_001.jpg", "この写真のタイトルを考えてください。"),
        ("artwork_002.jpg", "このアート作品の印象を述べてください。"),
        ("landscape_003.jpg", "この風景の特徴を説明してください。"),
        ("portrait_004.jpg", "この人物の表情から感情を読み取ってください。")
    ]
    
    results = []
    total_start = time.time()
    
    for image_id, prompt in tasks:
        print(f"\n処理中: {image_id}")
        
        result = client.infer_text_from_image_id(
            model_name="general_multimodal_model",
            text=prompt,
            image_id=image_id,
            image_base_url="http://pi.local/image",
            config={"max_length": 100}
        )
        
        if result.get('success'):
            results.append({
                'image_id': image_id,
                'prompt': prompt,
                'generated_text': result['generated_text'],
                'timing': result['timing']
            })
            print(f"✅ {result['generated_text']}")
        else:
            results.append({
                'image_id': image_id,
                'prompt': prompt,
                'error': result.get('error')
            })
            print(f"❌ エラー: {result.get('error')}")
    
    total_time = time.time() - total_start
    
    print(f"\n=== バッチ処理完了 ===")
    print(f"総処理時間: {total_time:.3f}秒")
    print(f"処理数: {len(tasks)}件")
    print(f"成功: {len([r for r in results if 'generated_text' in r])}件")

def example_web_integration():
    """Webシステム連携の例"""
    print("\n=== Webシステム連携例 ===")
    
    client = ImageIDInferenceClient()
    
    # Webアプリケーションでのユーザー操作をシミュレート
    user_sessions = [
        {
            "user_id": "user_001",
            "uploaded_image_id": "upload_12345.jpg",
            "question": "この商品の特徴を教えてください。"
        },
        {
            "user_id": "user_002", 
            "uploaded_image_id": "upload_67890.jpg",
            "question": "この画像から何がわかりますか？"
        }
    ]
    
    for session in user_sessions:
        print(f"\nユーザー: {session['user_id']}")
        print(f"画像ID: {session['uploaded_image_id']}")
        print(f"質問: {session['question']}")
        
        # Formベースのリクエスト（Webフォームからの送信をシミュレート）
        result = client.infer_text_from_image_id_form(
            model_name="product_analysis_model",
            text=session['question'],
            image_id=session['uploaded_image_id'],
            image_base_url="http://pi.local/image",
            maintain_aspect_ratio=True
        )
        
        if result.get('success'):
            print(f"AI応答: {result['generated_text']}")
            
            # Webアプリケーションでの後続処理をシミュレート
            response_data = {
                "user_id": session['user_id'],
                "ai_response": result['generated_text'],
                "processing_time": result['timing']['total_time'],
                "timestamp": time.time()
            }
            print(f"レスポンスデータ: {response_data}")
        else:
            print(f"エラーレスポンス: {result.get('error')}")

def main():
    """使用例の実行"""
    print("画像IDベース マルチモーダル推論クライアント")
    print("=" * 60)
    
    print("注意: 以下の例を実行するには準備が必要です：")
    print("1. PiVot-Serverが起動していること (python main.py)")
    print("2. 適切なマルチモーダルモデルが読み込まれていること")
    print("3. 画像サーバー (http://pi.local/image/) が稼働していること")
    print("4. 指定した画像IDが実際に存在すること")
    print()
    
    try:
        # 基本的な接続テスト
        client = ImageIDInferenceClient()
        
        # サーバー状態確認
        try:
            response = requests.get(f"{client.base_url}/")
            if response.status_code == 200:
                print("✅ サーバー接続OK")
                server_info = response.json()
                print(f"   NPU利用可能: {server_info.get('npu_available')}")
            else:
                print("❌ サーバー接続エラー")
                return
        except Exception as e:
            print(f"❌ サーバー接続失敗: {e}")
            return
        
        # 使用例の実行（実際の環境では以下のコメントを外す）
        # example_image_captioning()
        # example_visual_qa() 
        # example_batch_processing()
        # example_web_integration()
        
        print("\n🚀 クライアント準備完了！")
        print("実際の画像ID と モデルを指定して各関数を実行してください。")
        
    except Exception as e:
        print(f"実行エラー: {e}")

if __name__ == "__main__":
    main()