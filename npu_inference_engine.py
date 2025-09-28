"""
NPU Inference Engine
Intel NPUを使用した軽量推論エンジン
"""
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import openvino as ov

from npu_manager import npu_manager
from image_processor import default_image_processor, ImageProcessor
from text_processor import default_text_processor, TextProcessor
from text_output_processor import default_text_output_processor

logger = logging.getLogger(__name__)

class NPUInferenceEngine:
    """Intel NPUを使用した推論エンジンクラス"""
    
    def __init__(self):
        self.models: Dict[str, ov.CompiledModel] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.image_processor = default_image_processor
        self.text_processor = default_text_processor
        self.text_output_processor = default_text_output_processor
        
    def load_model(self, 
                   model_name: str,
                   model_path: str,
                   model_type: str = "vision",
                   input_shapes: Optional[Dict[str, Tuple]] = None,
                   **compile_config) -> bool:
        """
        モデルを読み込んでNPUにコンパイル
        
        Args:
            model_name: モデルの識別名
            model_path: OpenVINO IR形式のモデルファイルパス (.xml)
            model_type: モデルタイプ ("vision", "text", "multimodal")
            input_shapes: 入力テンソルの形状
            compile_config: NPU用のコンパイル設定
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # モデルをコンパイル
            compiled_model = npu_manager.compile_model(str(model_path), **compile_config)
            
            if compiled_model is None:
                logger.error(f"Failed to compile model: {model_name}")
                return False
            
            # モデル情報を保存
            self.models[model_name] = compiled_model
            
            # 入力/出力情報を取得
            input_info = {}
            output_info = {}
            
            for input_port in compiled_model.inputs:
                input_info[input_port.any_name] = {
                    'shape': input_port.shape,
                    'type': input_port.element_type
                }
            
            for output_port in compiled_model.outputs:
                output_info[output_port.any_name] = {
                    'shape': output_port.shape,
                    'type': output_port.element_type
                }
            
            self.model_configs[model_name] = {
                'type': model_type,
                'path': str(model_path),
                'inputs': input_info,
                'outputs': output_info,
                'input_shapes': input_shapes or {}
            }
            
            logger.info(f"Model loaded successfully: {model_name} ({model_type})")
            logger.info(f"Inputs: {list(input_info.keys())}")
            logger.info(f"Outputs: {list(output_info.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """モデルをアンロード"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                del self.model_configs[model_name]
                logger.info(f"Model unloaded: {model_name}")
                return True
            else:
                logger.warning(f"Model not found: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """モデル情報を取得"""
        return self.model_configs.get(model_name)
    
    def list_loaded_models(self) -> List[str]:
        """読み込み済みモデルのリストを取得"""
        return list(self.models.keys())
    
    def infer_image(self, 
                   model_name: str,
                   image_input: Union[bytes, str, np.ndarray],
                   preprocess_config: Optional[Dict] = None) -> Optional[Dict]:
        """
        画像推論を実行
        
        Args:
            model_name: 使用するモデル名
            image_input: 画像入力（バイト、ファイルパス、numpy配列）
            preprocess_config: 前処理設定
        """
        if model_name not in self.models:
            logger.error(f"Model not loaded: {model_name}")
            return None
            
        try:
            start_time = time.time()
            
            # 前処理設定のデフォルト値
            config = preprocess_config or {}
            maintain_aspect_ratio = config.get('maintain_aspect_ratio', False)
            
            # 画像前処理
            processed_image = self.image_processor.process_for_inference(
                image_input, 
                maintain_aspect_ratio=maintain_aspect_ratio
            )
            
            if processed_image is None:
                logger.error("Image preprocessing failed")
                return None
            
            preprocess_time = time.time() - start_time
            
            # 推論実行
            infer_start = time.time()
            compiled_model = self.models[model_name]
            
            # 入力名を取得（最初の入力を使用）
            input_name = list(compiled_model.inputs)[0].any_name
            
            # 推論実行
            result = compiled_model([processed_image])
            
            inference_time = time.time() - infer_start
            total_time = time.time() - start_time
            
            # 結果を整理
            outputs = {}
            for i, output_port in enumerate(compiled_model.outputs):
                outputs[output_port.any_name] = result[i]
            
            return {
                'success': True,
                'outputs': outputs,
                'model_name': model_name,
                'timing': {
                    'preprocess_time': preprocess_time,
                    'inference_time': inference_time,
                    'total_time': total_time
                },
                'input_shape': processed_image.shape,
                'device': npu_manager.npu_device
            }
            
        except Exception as e:
            logger.error(f"Image inference failed for {model_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def infer_text(self, 
                  model_name: str,
                  text_input: str,
                  preprocess_config: Optional[Dict] = None) -> Optional[Dict]:
        """
        テキスト推論を実行
        
        Args:
            model_name: 使用するモデル名
            text_input: テキスト入力
            preprocess_config: 前処理設定
        """
        if model_name not in self.models:
            logger.error(f"Model not loaded: {model_name}")
            return None
            
        try:
            start_time = time.time()
            
            # 前処理設定のデフォルト値
            config = preprocess_config or {}
            return_features = config.get('return_features', False)
            
            # テキスト前処理
            processed_text = self.text_processor.process_for_inference(
                text_input,
                return_features=return_features
            )
            
            preprocess_time = time.time() - start_time
            
            # 推論実行
            infer_start = time.time()
            compiled_model = self.models[model_name]
            
            # 入力データの準備
            inputs = {}
            for input_port in compiled_model.inputs:
                input_name = input_port.any_name
                if input_name in processed_text:
                    inputs[input_name] = processed_text[input_name]
                elif 'input_ids' in processed_text and input_name in ['input', 'input_ids', 'inputs']:
                    inputs[input_name] = processed_text['input_ids']
            
            if not inputs:
                logger.error("No matching inputs found for text model")
                return None
            
            # 推論実行
            result = compiled_model(inputs)
            
            inference_time = time.time() - infer_start
            total_time = time.time() - start_time
            
            # 結果を整理
            outputs = {}
            for i, output_port in enumerate(compiled_model.outputs):
                outputs[output_port.any_name] = result[i]
            
            return {
                'success': True,
                'outputs': outputs,
                'model_name': model_name,
                'timing': {
                    'preprocess_time': preprocess_time,
                    'inference_time': inference_time,
                    'total_time': total_time
                },
                'input_text_length': len(text_input),
                'device': npu_manager.npu_device
            }
            
        except Exception as e:
            logger.error(f"Text inference failed for {model_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def infer_multimodal(self,
                        model_name: str,
                        image_input: Union[bytes, str, np.ndarray],
                        text_input: str,
                        preprocess_config: Optional[Dict] = None) -> Optional[Dict]:
        """
        マルチモーダル推論を実行
        
        Args:
            model_name: 使用するモデル名
            image_input: 画像入力
            text_input: テキスト入力
            preprocess_config: 前処理設定
        """
        if model_name not in self.models:
            logger.error(f"Model not loaded: {model_name}")
            return None
            
        try:
            start_time = time.time()
            
            # 前処理設定
            config = preprocess_config or {}
            
            # 画像とテキストを並行処理
            processed_image = self.image_processor.process_for_inference(
                image_input,
                maintain_aspect_ratio=config.get('maintain_aspect_ratio', False)
            )
            
            processed_text = self.text_processor.process_for_inference(
                text_input,
                return_features=config.get('return_features', False)
            )
            
            if processed_image is None or not processed_text:
                logger.error("Multimodal preprocessing failed")
                return None
            
            preprocess_time = time.time() - start_time
            
            # 推論実行
            infer_start = time.time()
            compiled_model = self.models[model_name]
            
            # 入力データの準備
            inputs = {}
            
            # 画像入力のマッピング
            image_input_names = ['image', 'images', 'pixel_values', 'input_image']
            for input_port in compiled_model.inputs:
                input_name = input_port.any_name
                if any(name in input_name.lower() for name in image_input_names):
                    inputs[input_name] = processed_image
                    break
            
            # テキスト入力のマッピング
            for input_port in compiled_model.inputs:
                input_name = input_port.any_name
                if input_name in processed_text:
                    inputs[input_name] = processed_text[input_name]
            
            if len(inputs) < 2:  # 最低限画像とテキストの入力が必要
                logger.warning("Insufficient inputs for multimodal model")
            
            # 推論実行
            result = compiled_model(inputs)
            
            inference_time = time.time() - infer_start
            total_time = time.time() - start_time
            
            # 結果を整理
            outputs = {}
            for i, output_port in enumerate(compiled_model.outputs):
                outputs[output_port.any_name] = result[i]
            
            # テキスト出力を抽出（マルチモーダル推論用）
            generated_text = None
            model_config = self.model_configs.get(model_name, {})
            if model_config.get('type') == 'multimodal':
                try:
                    generated_text = self.text_output_processor.extract_text_from_multimodal_output(outputs)
                    if generated_text:
                        # プロンプトを考慮したフォーマット
                        generated_text = self.text_output_processor.format_conversation_output(
                            generated_text, text_input, add_context=True
                        )
                except Exception as e:
                    logger.warning(f"Text extraction failed: {e}")
            
            response_data = {
                'success': True,
                'outputs': outputs,
                'model_name': model_name,
                'timing': {
                    'preprocess_time': preprocess_time,
                    'inference_time': inference_time,
                    'total_time': total_time
                },
                'input_shapes': {
                    'image': processed_image.shape,
                    'text_length': len(text_input)
                },
                'device': npu_manager.npu_device
            }
            
            # 生成されたテキストがある場合は追加
            if generated_text:
                response_data['generated_text'] = generated_text
            
            return response_data
            
        except Exception as e:
            logger.error(f"Multimodal inference failed for {model_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def benchmark_model(self, 
                       model_name: str,
                       input_type: str = "random",
                       num_iterations: int = 10) -> Optional[Dict]:
        """モデルのベンチマークを実行"""
        if model_name not in self.models:
            logger.error(f"Model not loaded: {model_name}")
            return None
            
        try:
            compiled_model = self.models[model_name]
            model_config = self.model_configs[model_name]
            
            # ダミー入力の生成
            dummy_inputs = {}
            for input_port in compiled_model.inputs:
                input_name = input_port.any_name
                input_shape = input_port.shape
                
                if input_type == "random":
                    dummy_inputs[input_name] = np.random.randn(*input_shape).astype(np.float32)
                else:
                    dummy_inputs[input_name] = np.ones(input_shape, dtype=np.float32)
            
            # ウォームアップ
            for _ in range(3):
                compiled_model(dummy_inputs)
            
            # ベンチマーク実行
            times = []
            for i in range(num_iterations):
                start_time = time.time()
                result = compiled_model(dummy_inputs)
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 統計計算
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            return {
                'model_name': model_name,
                'model_type': model_config['type'],
                'device': npu_manager.npu_device,
                'iterations': num_iterations,
                'timing_stats': {
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'fps': 1.0 / mean_time
                },
                'input_shapes': {name: list(shape) for name, shape in 
                               [(port.any_name, port.shape) for port in compiled_model.inputs]}
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            return None

# グローバルインスタンス
npu_inference_engine = NPUInferenceEngine()