"""
Text Processing Module for NPU Inference
テキストの前処理とNPU用フォーマット変換
"""
import re
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)

class TextProcessor:
    """NPU推論用のテキスト処理クラス"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 use_fast_tokenizer: bool = True):
        """
        Args:
            model_name: 使用するトークナイザーのモデル名
            max_length: 最大テキスト長
            use_fast_tokenizer: 高速トークナイザーを使用するか
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self._initialize_tokenizer(use_fast_tokenizer)
        
    def _initialize_tokenizer(self, use_fast: bool) -> None:
        """トークナイザーを初期化"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=use_fast
            )
            logger.info(f"Tokenizer initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            # フォールバック：シンプルな文字レベル処理
            self.tokenizer = None
    
    def clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        if not text:
            return ""
            
        try:
            # 基本的なクリーニング
            text = text.strip()
            
            # 改行を空白に変換
            text = re.sub(r'\n+', ' ', text)
            
            # 複数の空白を単一の空白に
            text = re.sub(r'\s+', ' ', text)
            
            # 特殊文字の正規化
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            
            # 制御文字を削除
            text = ''.join(char for char in text if ord(char) >= 32)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    def tokenize_text(self, 
                     text: str, 
                     add_special_tokens: bool = True,
                     return_attention_mask: bool = True) -> Dict[str, np.ndarray]:
        """テキストをトークン化"""
        if not self.tokenizer:
            return self._simple_tokenize(text)
            
        try:
            # テキストをクリーニング
            clean_text = self.clean_text(text)
            
            # トークン化実行
            encoded = self.tokenizer(
                clean_text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=return_attention_mask,
                return_tensors='np'
            )
            
            result = {
                'input_ids': encoded['input_ids'],
            }
            
            if return_attention_mask and 'attention_mask' in encoded:
                result['attention_mask'] = encoded['attention_mask']
            
            if 'token_type_ids' in encoded:
                result['token_type_ids'] = encoded['token_type_ids']
            
            logger.info(f"Text tokenized: length={len(clean_text)}, tokens={encoded['input_ids'].shape}")
            return result
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return self._simple_tokenize(text)
    
    def _simple_tokenize(self, text: str) -> Dict[str, np.ndarray]:
        """フォールバック用の簡単なトークン化"""
        try:
            clean_text = self.clean_text(text)
            
            # 文字レベルの簡単なエンコード（ASCII基準）
            char_ids = [ord(char) for char in clean_text[:self.max_length]]
            
            # パディング
            if len(char_ids) < self.max_length:
                char_ids.extend([0] * (self.max_length - len(char_ids)))
            
            input_ids = np.array([char_ids], dtype=np.int64)
            attention_mask = np.array([[1 if x > 0 else 0 for x in char_ids]], dtype=np.int64)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            logger.error(f"Simple tokenization failed: {e}")
            # 最終フォールバック
            return {
                'input_ids': np.zeros((1, self.max_length), dtype=np.int64),
                'attention_mask': np.zeros((1, self.max_length), dtype=np.int64)
            }
    
    def encode_text_pair(self, 
                        text_a: str, 
                        text_b: str,
                        max_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        """テキストペアをエンコード（質問応答等で使用）"""
        if not self.tokenizer:
            # 簡単な結合処理
            combined_text = f"{text_a} [SEP] {text_b}"
            return self.tokenize_text(combined_text)
            
        try:
            max_len = max_length or self.max_length
            
            encoded = self.tokenizer(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np'
            )
            
            result = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            }
            
            if 'token_type_ids' in encoded:
                result['token_type_ids'] = encoded['token_type_ids']
            
            return result
            
        except Exception as e:
            logger.error(f"Text pair encoding failed: {e}")
            # フォールバック
            combined_text = f"{text_a} {text_b}"
            return self.tokenize_text(combined_text)
    
    def decode_tokens(self, token_ids: Union[List[int], np.ndarray]) -> str:
        """トークンIDをテキストにデコード"""
        if not self.tokenizer:
            # 簡単な文字レベルデコード
            try:
                if isinstance(token_ids, np.ndarray):
                    token_ids = token_ids.flatten().tolist()
                return ''.join([chr(int(tid)) for tid in token_ids if tid > 0])
            except:
                return ""
                
        try:
            if isinstance(token_ids, np.ndarray):
                token_ids = token_ids.flatten().tolist()
                
            decoded_text = self.tokenizer.decode(
                token_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            return decoded_text.strip()
            
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            return ""
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """テキストから基本的な特徴を抽出"""
        try:
            clean_text = self.clean_text(text)
            
            features = {
                'length': len(clean_text),
                'word_count': len(clean_text.split()),
                'sentence_count': len([s for s in clean_text.split('.') if s.strip()]),
                'has_question': '?' in clean_text,
                'has_exclamation': '!' in clean_text,
                'uppercase_ratio': sum(1 for c in clean_text if c.isupper()) / max(len(clean_text), 1),
                'digit_count': sum(1 for c in clean_text if c.isdigit()),
                'special_char_count': sum(1 for c in clean_text if not c.isalnum() and not c.isspace())
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def process_for_inference(self, 
                            text: str,
                            return_features: bool = False) -> Dict[str, np.ndarray]:
        """推論用の完全なテキスト処理パイプライン"""
        try:
            # トークン化
            tokens = self.tokenize_text(text)
            
            # 特徴抽出（オプション）
            if return_features:
                features = self.extract_features(text)
                tokens['text_features'] = features
            
            return tokens
            
        except Exception as e:
            logger.error(f"Full text processing failed: {e}")
            return {
                'input_ids': np.zeros((1, self.max_length), dtype=np.int64),
                'attention_mask': np.zeros((1, self.max_length), dtype=np.int64)
            }

# デフォルトのグローバルインスタンス
default_text_processor = TextProcessor()