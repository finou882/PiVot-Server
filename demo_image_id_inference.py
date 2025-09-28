"""
Image ID Inference Integration Demo
画像IDベース推論の統合デモ
"""
import time
import subprocess
import requests
import json
from typing import Dict, Any
from image_id_client import ImageIDInferenceClient

def check_service(url: str, service_name: str) -> bool:
    """サービスの稼働状態をチェック"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is running")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} is not accessible: {e}")
        return False

def demonstrate_image_id_inference():
    """画像IDベース推論のデモンストレーション"""
    print("🎯 画像IDベース マルチモーダル推論デモ")
    print("=" * 60)
    
    # 1. サービス稼働状態確認
    print("1. サービス稼働確認...")
    
    pivot_server_ok = check_service("http://localhost:8000", "PiVot-Server")
    image_server_ok = check_service("http://localhost:8001", "Mock Image Server")
    
    if not pivot_server_ok:
        print("\n💡 PiVot-Serverを起動してください:")
        print("   python main.py")
        print()
    
    if not image_server_ok:
        print("\n💡 Mock Image Serverを起動してください:")
        print("   python mock_image_server.py")
        print()
    
    if not (pivot_server_ok and image_server_ok):
        print("❌ 必要なサービスが起動していません。")
        return False
    
    print()
    
    # 2. クライアント初期化
    client = ImageIDInferenceClient(base_url="http://localhost:8000")
    
    # 3. 利用可能なサンプル画像を取得
    print("2. 利用可能なサンプル画像確認...")
    try:
        samples_response = requests.get("http://localhost:8001/samples")
        samples = samples_response.json()
        sample_images = samples["available_samples"][:3]  # 最初の3つを使用
        print(f"   サンプル画像: {sample_images}")
    except Exception as e:
        print(f"❌ サンプル画像取得失敗: {e}")
        sample_images = ["sample_001.jpg", "test_image.png", "demo_photo.jpg"]
    
    print()
    
    # 4. 様々なシナリオでデモ実行
    demo_scenarios = [
        {
            "name": "画像キャプション生成",
            "image_id": sample_images[0],
            "prompts": [
                "この画像に何が写っていますか？",
                "この画像の特徴を教えてください。"
            ]
        },
        {
            "name": "視覚的質問応答",
            "image_id": sample_images[1] if len(sample_images) > 1 else sample_images[0],
            "prompts": [
                "この画像の色は何色ですか？",
                "この画像から受ける印象を教えてください。"
            ]
        },
        {
            "name": "創作的テキスト生成",
            "image_id": sample_images[2] if len(sample_images) > 2 else sample_images[0],
            "prompts": [
                "この画像を見て短い物語を作ってください。",
                "この画像にタイトルを付けるとしたら何ですか？"
            ]
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(demo_scenarios, 3):
        print(f"{i}. {scenario['name']}デモ...")
        print(f"   使用画像: {scenario['image_id']}")
        
        for j, prompt in enumerate(scenario['prompts'], 1):
            print(f"\n   --- {scenario['name']} {j} ---")
            print(f"   プロンプト: {prompt}")
            
            # JSON版とForm版の両方をテスト
            test_methods = [
                ("JSON", client.infer_text_from_image_id),
                ("Form", client.infer_text_from_image_id_form)
            ]
            
            for method_name, method_func in test_methods:
                print(f"   [{method_name}版] ", end="")
                
                start_time = time.time()
                result = method_func(
                    model_name="demo_multimodal_model",  # 実際の環境では適切なモデル名を指定
                    text=prompt,
                    image_id=scenario['image_id'],
                    image_base_url="http://localhost:8001/image",
                    maintain_aspect_ratio=True,
                    config={"max_length": 150}
                )
                end_time = time.time()
                
                if result.get('success'):
                    print("✅ 成功")
                    print(f"      生成テキスト: {result.get('generated_text', 'N/A')}")
                    print(f"      推論時間: {result.get('timing', {}).get('total_time', 'N/A'):.3f}秒")
                    print(f"      画像取得時間: {result.get('timing', {}).get('image_fetch_time', 'N/A'):.3f}秒")
                    print(f"      使用デバイス: {result.get('device', 'N/A')}")
                    
                    results.append({
                        'scenario': scenario['name'],
                        'method': method_name,
                        'success': True,
                        'response_time': end_time - start_time,
                        'generated_text': result.get('generated_text'),
                        'timing': result.get('timing', {})
                    })
                else:
                    print("❌ 失敗")
                    error_msg = result.get('error', 'Unknown error')
                    print(f"      エラー: {error_msg}")
                    
                    results.append({
                        'scenario': scenario['name'],
                        'method': method_name,
                        'success': False,
                        'error': error_msg,
                        'response_time': end_time - start_time
                    })
                
                # JSON版とForm版の間に短い待機
                time.sleep(0.5)
            
            print()
    
    # 5. 結果サマリー
    print(f"{len(demo_scenarios) + 2}. デモ結果サマリー")
    print("=" * 40)
    
    successful_requests = [r for r in results if r['success']]
    failed_requests = [r for r in results if not r['success']]
    
    print(f"総リクエスト数: {len(results)}")
    print(f"成功: {len(successful_requests)} 件")
    print(f"失敗: {len(failed_requests)} 件")
    
    if successful_requests:
        avg_response_time = sum(r['response_time'] for r in successful_requests) / len(successful_requests)
        print(f"平均レスポンス時間: {avg_response_time:.3f}秒")
    
    if failed_requests:
        print("\n❌ 失敗したリクエスト:")
        for req in failed_requests:
            print(f"   {req['scenario']} ({req['method']}): {req['error']}")
    
    print("\n🎉 デモ完了！")
    
    return len(successful_requests) > 0

def show_usage_examples():
    """実用例の表示"""
    print("\n📖 実用例")
    print("=" * 40)
    
    examples = [
        {
            "title": "商品説明生成",
            "code": '''
# 商品画像から説明文を生成
result = client.infer_text_from_image_id(
    model_name="product_description_model",
    text="この商品の特徴と魅力を説明してください。",
    image_id="product_12345.jpg",
    image_base_url="https://shop.example.com/images"
)
print(result['generated_text'])
'''
        },
        {
            "title": "医療画像診断支援", 
            "code": '''
# 医療画像の所見生成
result = client.infer_text_from_image_id(
    model_name="medical_analysis_model",
    text="この画像から観察される所見を教えてください。",
    image_id="xray_001.jpg",
    image_base_url="https://hospital.example.com/images"
)
print(result['generated_text'])
'''
        },
        {
            "title": "教育コンテンツ生成",
            "code": '''
# 教材画像から説明文生成
result = client.infer_text_from_image_id(
    model_name="educational_model", 
    text="この図表を小学生にもわかるように説明してください。",
    image_id="diagram_biology_001.jpg",
    image_base_url="https://edu.example.com/images"
)
print(result['generated_text'])
'''
        }
    ]
    
    for example in examples:
        print(f"\n📌 {example['title']}")
        print(example['code'])

def main():
    """メイン実行"""
    success = demonstrate_image_id_inference()
    
    if success:
        show_usage_examples()
        
        print("\n🔧 開発者向け情報")
        print("=" * 40)
        print("APIエンドポイント:")
        print("  JSON版: POST /infer/text-from-image-id")
        print("  Form版: POST /infer/text-from-image-id/form")
        print()
        print("必要なパラメータ:")
        print("  - model_name: 使用するマルチモーダルモデル名")
        print("  - text: テキストプロンプト")
        print("  - image_id: 画像のID")
        print("  - image_base_url: 画像サーバーのベースURL")
        print()
        print("レスポンス:")
        print("  - success: 成功/失敗")
        print("  - generated_text: 生成されたテキスト") 
        print("  - timing: 実行時間の詳細")
        print("  - device: 使用されたデバイス")
    else:
        print("\n⚠️ デモが完了しませんでした。")
        print("必要なサービスを起動してから再実行してください。")

if __name__ == "__main__":
    main()