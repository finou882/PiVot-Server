"""
Image Processing Module for NPU Inference
画像の前処理とNPU用フォーマット変換
"""
import io
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """NPU推論用の画像処理クラス"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_size: NPUモデル用のターゲットサイズ (width, height)
        """
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406])  # ImageNet標準
        self.std = np.array([0.229, 0.224, 0.225])   # ImageNet標準
        
    def load_image_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """バイトデータから画像を読み込み"""
        try:
            # PILで画像を読み込み
            image = Image.open(io.BytesIO(image_bytes))
            
            # RGBAやグレースケールをRGBに変換
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # NumPy配列に変換
            image_array = np.array(image)
            
            logger.info(f"Image loaded: shape={image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to load image from bytes: {e}")
            return None
    
    def load_image_from_file(self, image_path: str) -> Optional[np.ndarray]:
        """ファイルから画像を読み込み"""
        try:
            # OpenCVで読み込み（BGR -> RGB変換）
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image from {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Image loaded from file: {image_path}, shape={image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image from file {image_path}: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """NPU推論用に画像を前処理"""
        try:
            # リサイズ
            image_resized = cv2.resize(image, self.target_size)
            
            # float32に変換し、0-1の範囲に正規化
            image_float = image_resized.astype(np.float32) / 255.0
            
            # ImageNet標準化（オプション）
            if normalize:
                image_float = (image_float - self.mean) / self.std
            
            # CHW形式に変換 (Channels, Height, Width)
            image_chw = np.transpose(image_float, (2, 0, 1))
            
            # バッチ次元を追加 (N, C, H, W)
            image_batch = np.expand_dims(image_chw, axis=0)
            
            logger.info(f"Image preprocessed: {image_batch.shape}")
            return image_batch
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def postprocess_image(self, processed_image: np.ndarray) -> np.ndarray:
        """前処理された画像を表示用に戻す"""
        try:
            # バッチ次元を削除
            if len(processed_image.shape) == 4:
                image = processed_image[0]
            else:
                image = processed_image
            
            # CHW -> HWC形式に変換
            if len(image.shape) == 3 and image.shape[0] <= 4:  # チャンネル数をチェック
                image = np.transpose(image, (1, 2, 0))
            
            # 正規化を戻す
            image = image * self.std + self.mean
            
            # 0-255の範囲に戻す
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Image postprocessing failed: {e}")
            return processed_image
    
    def resize_maintain_aspect_ratio(self, 
                                   image: np.ndarray, 
                                   target_size: Tuple[int, int],
                                   fill_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        """アスペクト比を保ってリサイズ（パディング付き）"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # スケール計算
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # リサイズ
            resized = cv2.resize(image, (new_w, new_h))
            
            # パディングで目標サイズに調整
            result = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
            
            # 中央配置
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return result
            
        except Exception as e:
            logger.error(f"Aspect ratio resize failed: {e}")
            return cv2.resize(image, target_size)  # フォールバック
    
    def process_for_inference(self, 
                            image_input: Union[bytes, str, np.ndarray],
                            maintain_aspect_ratio: bool = False) -> Optional[np.ndarray]:
        """推論用の完全な画像処理パイプライン"""
        try:
            # 入力形式に応じて画像を読み込み
            if isinstance(image_input, bytes):
                image = self.load_image_from_bytes(image_input)
            elif isinstance(image_input, str):
                image = self.load_image_from_file(image_input)
            elif isinstance(image_input, np.ndarray):
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            if image is None:
                return None
            
            # アスペクト比保持リサイズまたは通常リサイズ
            if maintain_aspect_ratio:
                image = self.resize_maintain_aspect_ratio(image, self.target_size)
            
            # 前処理実行
            processed_image = self.preprocess_image(image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Full image processing failed: {e}")
            return None

# デフォルトのグローバルインスタンス
default_image_processor = ImageProcessor()