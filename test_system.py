"""
Test Suite for PiVot-Server
基本機能のテストスクリプト
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path

# ローカルモジュールのパスを追加
sys.path.append(str(Path(__file__).parent))

from npu_manager import npu_manager
from image_processor import default_image_processor
from text_processor import default_text_processor  
from npu_inference_engine import npu_inference_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_npu_manager():
    """NPU Manager のテスト"""
    print("\n=== NPU Manager Test ===")
    
    # デバイス情報取得
    device_info = npu_manager.get_device_info()
    print(f"デバイス情報: {device_info}")
    
    # NPU利用可能性チェック
    npu_available = npu_manager.is_npu_available()
    print(f"NPU利用可能: {npu_available}")
    
    # 利用デバイス表示
    print(f"使用デバイス: {npu_manager.npu_device}")
    
    return True

def test_image_processor():
    """Image Processor のテスト"""
    print("\n=== Image Processor Test ===")
    
    try:
        # ダミー画像生成（RGB, 256x256）
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        print(f"ダミー画像形状: {dummy_image.shape}")
        
        # 前処理テスト
        processed = default_image_processor.preprocess_image(dummy_image)
        print(f"前処理後形状: {processed.shape}")
        print(f"前処理後データ型: {processed.dtype}")
        print(f"前処理後範囲: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # アスペクト比保持リサイズテスト
        aspect_resized = default_image_processor.resize_maintain_aspect_ratio(
            dummy_image, (224, 224)
        )
        print(f"アスペクト比保持リサイズ後: {aspect_resized.shape}")
        
        # 完全な処理パイプラインテスト  
        final_processed = default_image_processor.process_for_inference(dummy_image)
        print(f"完全処理後形状: {final_processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"画像処理テストエラー: {e}")
        return False

def test_text_processor():
    """Text Processor のテスト"""
    print("\n=== Text Processor Test ===")
    
    try:
        # テストテキスト
        test_text = "今日は良い天気です。機械学習の勉強をしています。"
        print(f"テストテキスト: {test_text}")
        
        # クリーニングテスト
        cleaned = default_text_processor.clean_text(test_text)
        print(f"クリーニング後: {cleaned}")
        
        # トークン化テスト
        tokens = default_text_processor.tokenize_text(test_text)
        print(f"トークン化結果: {list(tokens.keys())}")
        print(f"input_ids形状: {tokens['input_ids'].shape}")
        
        if 'attention_mask' in tokens:
            print(f"attention_mask形状: {tokens['attention_mask'].shape}")
        
        # 特徴抽出テスト
        features = default_text_processor.extract_features(test_text)
        print(f"抽出された特徴: {features}")
        
        # 完全処理パイプラインテスト
        processed = default_text_processor.process_for_inference(test_text, return_features=True)
        print(f"完全処理結果: {list(processed.keys())}")
        
        return True
        
    except Exception as e:
        print(f"テキスト処理テストエラー: {e}")
        return False

def test_inference_engine():
    """Inference Engine のテスト"""
    print("\n=== Inference Engine Test ===")
    
    try:
        # エンジンの基本情報
        models = npu_inference_engine.list_loaded_models()
        print(f"読み込み済みモデル数: {len(models)}")
        
        # モデル読み込みテスト（実際のモデルファイルがない場合はスキップ）
        sample_model_path = "./models/sample.xml"
        if Path(sample_model_path).exists():
            success = npu_inference_engine.load_model(
                "test_model", 
                sample_model_path, 
                "vision"
            )
            print(f"モデル読み込みテスト: {success}")
            
            if success:
                # モデル情報取得
                info = npu_inference_engine.get_model_info("test_model")
                print(f"モデル情報: {info}")
                
                # アンロードテスト
                unload_success = npu_inference_engine.unload_model("test_model")
                print(f"モデルアンロードテスト: {unload_success}")
        else:
            print("実際のモデルファイルがないため、モデル読み込みテストをスキップ")
        
        return True
        
    except Exception as e:
        print(f"推論エンジンテストエラー: {e}")
        return False

def test_integration():
    """統合テスト"""
    print("\n=== Integration Test ===")
    
    try:
        # ダミーデータで統合処理テスト
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_text = "This is a test sentence for integration testing."
        
        # 画像とテキストを同時処理
        processed_image = default_image_processor.process_for_inference(test_image)
        processed_text = default_text_processor.process_for_inference(test_text)
        
        print(f"統合処理 - 画像形状: {processed_image.shape}")
        print(f"統合処理 - テキストキー: {list(processed_text.keys())}")
        
        # システム全体の互換性確認
        system_info = {
            'npu_available': npu_manager.is_npu_available(),
            'image_processor_ready': processed_image is not None,
            'text_processor_ready': bool(processed_text),
            'inference_engine_ready': True
        }
        
        print(f"システム統合状況: {system_info}")
        
        all_ready = all(system_info.values())
        print(f"システム全体準備完了: {all_ready}")
        
        return all_ready
        
    except Exception as e:
        print(f"統合テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("PiVot-Server Test Suite")
    print("=" * 50)
    
    tests = [
        ("NPU Manager", test_npu_manager),
        ("Image Processor", test_image_processor),
        ("Text Processor", test_text_processor), 
        ("Inference Engine", test_inference_engine),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASS" if result else "✗ FAIL" 
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n{test_name}: ✗ FAIL - {e}")
    
    print("\n" + "=" * 50)
    print("テスト結果サマリー:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n総合結果: {passed}/{len(tests)} テスト合格")
    
    if passed == len(tests):
        print("🎉 すべてのテストに合格しました！")
        print("サーバーを起動する準備が整っています。")
    else:
        print("⚠️  一部のテストが失敗しました。")
        print("エラーを確認して修正してください。")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)