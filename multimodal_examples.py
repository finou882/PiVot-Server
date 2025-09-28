"""
Multimodal Text Generation Examples
画像とテキストを使った文章生成の使用例
"""
import requests
import json
import base64
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import io

class MultimodalTextGenerator:
    """マルチモーダルテキスト生成クライアント"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def generate_text_from_image_and_prompt(self, 
                                          model_name: str,
                                          image_path: str, 
                                          text_prompt: str,
                                          config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        画像とテキストプロンプトから文章を生成
        
        Args:
            model_name: 使用するモデル名（例: "clip", "blip", "flamingo"）
            image_path: 画像ファイルパス
            text_prompt: テキストプロンプト
            config: 前処理設定
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {
                    'model_name': model_name,
                    'text': text_prompt,
                    'maintain_aspect_ratio': True,
                    'return_features': False
                }
                
                if config:
                    data['config'] = json.dumps(config)
                
                response = requests.post(
                    f"{self.base_url}/infer/multimodal", 
                    files=files, 
                    data=data
                )
                
                return response.json()
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_generated_text(self, inference_result: Dict) -> str:
        """推論結果からテキストを抽出"""
        if not inference_result.get('success'):
            return f"Error: {inference_result.get('error', 'Unknown error')}"
        
        outputs = inference_result.get('outputs', {})
        
        # 一般的な出力キー名を試行
        text_keys = ['generated_text', 'text_output', 'caption', 'description', 
                    'logits', 'text_logits', 'output', 'predictions']
        
        for key in text_keys:
            if key in outputs:
                output = outputs[key]
                
                # リスト形式の場合は最初の要素を取得
                if isinstance(output, list):
                    output = output[0] if output else ""
                
                # NumPy配列の場合は適切に処理
                if isinstance(output, (list, tuple)) and len(output) > 0:
                    # トークンIDの場合はテキストに変換（簡易版）
                    if all(isinstance(x, (int, float)) for x in output):
                        # この部分は実際のモデルの出力形式に応じて調整が必要
                        return f"Token IDs: {output[:10]}..." if len(output) > 10 else f"Token IDs: {output}"
                    else:
                        return str(output)
                
                return str(output)
        
        # デフォルト：すべての出力を表示
        return f"Raw outputs: {outputs}"

# 使用例の関数
def example_image_captioning():
    """画像キャプション生成の例"""
    print("=== 画像キャプション生成例 ===")
    
    generator = MultimodalTextGenerator()
    
    # 画像の説明を生成
    result = generator.generate_text_from_image_and_prompt(
        model_name="clip_caption_model",  # 実際のモデル名に置き換え
        image_path="sample_image.jpg",    # 実際の画像パスに置き換え
        text_prompt="この画像を詳しく説明してください。",
        config={
            "maintain_aspect_ratio": True,
            "max_length": 100
        }
    )
    
    if result['success']:
        generated_text = generator.extract_generated_text(result)
        print(f"生成されたキャプション: {generated_text}")
        print(f"推論時間: {result['timing']['total_time']:.3f}秒")
    else:
        print(f"エラー: {result['error']}")

def example_visual_question_answering():
    """視覚的質問応答の例"""
    print("=== 視覚的質問応答例 ===")
    
    generator = MultimodalTextGenerator()
    
    # 画像に関する質問に答える
    questions = [
        "この画像に何が写っていますか？",
        "写っている人は何をしていますか？", 
        "この場面はどこで撮影されたと思いますか？",
        "画像の雰囲気を一言で表現してください。"
    ]
    
    for question in questions:
        result = generator.generate_text_from_image_and_prompt(
            model_name="vqa_model",  # 実際のVQAモデル名に置き換え
            image_path="sample_image.jpg",
            text_prompt=question
        )
        
        if result['success']:
            answer = generator.extract_generated_text(result)
            print(f"Q: {question}")
            print(f"A: {answer}")
            print("-" * 50)
        else:
            print(f"Q: {question}")
            print(f"A: エラー - {result['error']}")
            print("-" * 50)

def example_story_generation():
    """画像からストーリー生成の例"""
    print("=== 画像ベースストーリー生成例 ===")
    
    generator = MultimodalTextGenerator()
    
    # 画像を基にした短編ストーリー生成
    result = generator.generate_text_from_image_and_prompt(
        model_name="story_generation_model",
        image_path="sample_image.jpg",
        text_prompt="この画像を見て、想像力豊かな短編ストーリーを書いてください。登場人物の気持ちや背景も含めて。",
        config={
            "max_length": 300,
            "temperature": 0.8  # 創造性を高める
        }
    )
    
    if result['success']:
        story = generator.extract_generated_text(result)
        print("生成されたストーリー:")
        print(story)
    else:
        print(f"ストーリー生成エラー: {result['error']}")

def example_image_to_code():
    """画像からコード生成の例"""
    print("=== 画像からコード生成例 ===")
    
    generator = MultimodalTextGenerator()
    
    # UI/図解からコード生成
    result = generator.generate_text_from_image_and_prompt(
        model_name="code_generation_model",
        image_path="ui_mockup.jpg",  # UI設計図など
        text_prompt="この画像のUIデザインを実装するためのHTML/CSSコードを生成してください。",
        config={
            "maintain_aspect_ratio": True,
            "focus_on_structure": True
        }
    )
    
    if result['success']:
        code = generator.extract_generated_text(result)
        print("生成されたコード:")
        print(code)
    else:
        print(f"コード生成エラー: {result['error']}")

def example_batch_processing():
    """バッチ処理の例"""
    print("=== バッチ処理例 ===")
    
    generator = MultimodalTextGenerator()
    
    # 複数の画像を順次処理
    image_tasks = [
        ("image1.jpg", "この画像のタイトルを考えてください。"),
        ("image2.jpg", "この画像の感情的な印象を表現してください。"),
        ("image3.jpg", "この画像で起きていることを時系列で説明してください。")
    ]
    
    results = []
    for image_path, prompt in image_tasks:
        result = generator.generate_text_from_image_and_prompt(
            model_name="general_multimodal_model",
            image_path=image_path,
            text_prompt=prompt
        )
        
        if result['success']:
            text = generator.extract_generated_text(result)
            results.append({
                'image': image_path,
                'prompt': prompt,
                'generated_text': text,
                'processing_time': result['timing']['total_time']
            })
        else:
            results.append({
                'image': image_path,
                'prompt': prompt,
                'error': result['error']
            })
    
    # 結果を表示
    for i, result in enumerate(results, 1):
        print(f"\n--- タスク {i} ---")
        print(f"画像: {result['image']}")
        print(f"プロンプト: {result['prompt']}")
        
        if 'generated_text' in result:
            print(f"生成テキスト: {result['generated_text']}")
            print(f"処理時間: {result['processing_time']:.3f}秒")
        else:
            print(f"エラー: {result['error']}")

def main():
    """使用例実行"""
    print("マルチモーダルテキスト生成の使用例")
    print("=" * 60)
    
    # 注意：実際に実行するには適切なモデルと画像ファイルが必要
    print("注意: これらの例を実行するには以下が必要です：")
    print("1. 適切なマルチモーダルモデル（CLIP, BLIP, Flamingo等）")
    print("2. 実際の画像ファイル")
    print("3. サーバーの起動")
    print()
    
    try:
        # 実際の例（コメントアウトされている部分を有効化して使用）
        # example_image_captioning()
        # example_visual_question_answering()
        # example_story_generation()
        # example_image_to_code()
        # example_batch_processing()
        
        print("使用例の準備が完了しました。")
        print("実際のモデルと画像を用意して各関数のコメントアウトを外してください。")
        
    except Exception as e:
        print(f"実行エラー: {e}")

if __name__ == "__main__":
    main()