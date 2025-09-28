"""
PiVot-Server Usage Examples
使用例とサンプルコード
"""
import requests
import json
import time
from pathlib import Path

class PiVotClient:
    """PiVot-Serverクライアントクラス"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def get_system_status(self):
        """システムステータスを取得"""
        response = requests.get(f"{self.base_url}/system/status")
        return response.json()
    
    def load_model(self, model_name: str, model_path: str, model_type: str = "vision"):
        """モデルを読み込み"""
        data = {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': model_type
        }
        response = requests.post(f"{self.base_url}/models/load", data=data)
        return response.json()
    
    def unload_model(self, model_name: str):
        """モデルをアンロード"""
        response = requests.delete(f"{self.base_url}/models/{model_name}")
        return response.json()
    
    def list_models(self):
        """読み込み済みモデル一覧を取得"""
        response = requests.get(f"{self.base_url}/models")
        return response.json()
    
    def infer_image(self, 
                   model_name: str, 
                   image_path: str,
                   maintain_aspect_ratio: bool = False,
                   config: dict = None):
        """画像推論を実行"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'model_name': model_name,
                'maintain_aspect_ratio': maintain_aspect_ratio
            }
            if config:
                data['config'] = json.dumps(config)
            
            response = requests.post(f"{self.base_url}/infer/image", files=files, data=data)
            return response.json()
    
    def infer_text(self, 
                  model_name: str, 
                  text: str,
                  return_features: bool = False,
                  config: dict = None):
        """テキスト推論を実行"""
        data = {
            'model_name': model_name,
            'text': text,
            'return_features': return_features
        }
        if config:
            data['config'] = json.dumps(config)
            
        response = requests.post(f"{self.base_url}/infer/text", data=data)
        return response.json()
    
    def infer_multimodal(self,
                        model_name: str,
                        image_path: str,
                        text: str,
                        maintain_aspect_ratio: bool = False,
                        return_features: bool = False,
                        config: dict = None):
        """マルチモーダル推論を実行"""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'model_name': model_name,
                'text': text,
                'maintain_aspect_ratio': maintain_aspect_ratio,
                'return_features': return_features
            }
            if config:
                data['config'] = json.dumps(config)
            
            response = requests.post(f"{self.base_url}/infer/multimodal", files=files, data=data)
            return response.json()
    
    def benchmark_model(self, model_name: str, iterations: int = 10):
        """モデルのベンチマークを実行"""
        data = {'iterations': iterations, 'input_type': 'random'}
        response = requests.post(f"{self.base_url}/benchmark/{model_name}", json=data)
        return response.json()

def main():
    """メイン実行例"""
    client = PiVotClient()
    
    # 1. システムステータス確認
    print("=== システムステータス ===")
    status = client.get_system_status()
    print(f"NPU利用可能: {status.get('npu_available')}")
    print(f"デバイス: {status.get('device_info', {}).get('selected_device')}")
    print()
    
    # 2. モデル読み込み例（実際のモデルファイルが必要）
    print("=== モデル読み込み例 ===")
    # model_result = client.load_model(
    #     model_name="sample_vision_model",
    #     model_path="./models/resnet50.xml",
    #     model_type="vision"
    # )
    # print(f"モデル読み込み: {model_result}")
    
    # 3. 読み込み済みモデル一覧
    print("=== 読み込み済みモデル ===")
    models = client.list_models()
    print(f"モデル一覧: {models}")
    print()
    
    # 4. 画像推論例（実際の画像ファイルとモデルが必要）
    # if models and Path("sample_image.jpg").exists():
    #     print("=== 画像推論例 ===")
    #     result = client.infer_image(
    #         model_name=models[0],
    #         image_path="sample_image.jpg",
    #         maintain_aspect_ratio=True
    #     )
    #     print(f"推論成功: {result.get('success')}")
    #     if result.get('success'):
    #         print(f"実行時間: {result['timing']['total_time']:.3f}秒")
    #         print(f"デバイス: {result.get('device')}")
    #     print()
    
    # 5. テキスト推論例
    # if models:
    #     print("=== テキスト推論例 ===")
    #     result = client.infer_text(
    #         model_name=models[0],  # 実際はテキストモデルを指定
    #         text="今日は良い天気です。機械学習の勉強をしています。",
    #         return_features=True
    #     )
    #     print(f"推論成功: {result.get('success')}")
    #     if result.get('success'):
    #         print(f"実行時間: {result['timing']['total_time']:.3f}秒")
    #     print()
    
    # 6. ベンチマーク例
    # if models:
    #     print("=== ベンチマーク例 ===")
    #     benchmark = client.benchmark_model(models[0], iterations=50)
    #     if 'timing_stats' in benchmark:
    #         stats = benchmark['timing_stats']
    #         print(f"平均推論時間: {stats['mean_time']:.3f}秒")
    #         print(f"FPS: {stats['fps']:.1f}")
    #         print(f"最小時間: {stats['min_time']:.3f}秒")
    #         print(f"最大時間: {stats['max_time']:.3f}秒")
    
    print("使用例の実行が完了しました。")
    print("実際の推論を実行するには、OpenVINOモデルファイルと入力データが必要です。")

if __name__ == "__main__":
    main()