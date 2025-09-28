"""
Text Output Processing Utilities
テキスト出力の後処理ユーティリティ
"""
import re
import numpy as np
from typing import Union, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TextOutputProcessor:
    """推論結果からのテキスト出力処理クラス"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def decode_token_ids(self, token_ids: Union[List[int], np.ndarray], 
                        skip_special_tokens: bool = True) -> str:
        """トークンIDをテキストにデコード"""
        try:
            if self.tokenizer:
                # Transformersトークナイザーを使用
                if isinstance(token_ids, np.ndarray):
                    token_ids = token_ids.flatten().tolist()
                
                # パディングトークンや特殊トークンを除去
                if skip_special_tokens and hasattr(self.tokenizer, 'pad_token_id'):
                    token_ids = [t for t in token_ids if t != self.tokenizer.pad_token_id]
                
                text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                return text.strip()
            else:
                # 簡単な文字ベースデコード
                if isinstance(token_ids, np.ndarray):
                    token_ids = token_ids.flatten().tolist()
                
                # ASCII範囲の文字のみ
                chars = [chr(int(t)) for t in token_ids if 32 <= int(t) <= 126]
                return ''.join(chars).strip()
                
        except Exception as e:
            logger.error(f"Token decoding failed: {e}")
            return f"[Decoding Error: {e}]"
    
    def process_logits_to_text(self, logits: np.ndarray, 
                              top_k: int = 5,
                              temperature: float = 1.0) -> str:
        """ロジットから確率的にテキストを生成"""
        try:
            if len(logits.shape) == 3:  # (batch, seq_len, vocab_size)
                logits = logits[0]  # 最初のバッチを使用
            
            if len(logits.shape) == 2:  # (seq_len, vocab_size)
                logits = logits[-1]  # 最後のトークンを使用
            
            # 温度調整
            logits = logits / temperature
            
            # Top-kサンプリング
            top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
            top_k_logits = logits[top_k_indices]
            
            # ソフトマックス
            exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # サンプリング
            selected_idx = np.random.choice(top_k, p=probabilities)
            selected_token_id = top_k_indices[selected_idx]
            
            return self.decode_token_ids([selected_token_id])
            
        except Exception as e:
            logger.error(f"Logits to text conversion failed: {e}")
            return "[Generation Error]"
    
    def clean_generated_text(self, text: str) -> str:
        """生成されたテキストをクリーニング"""
        if not text:
            return ""
        
        try:
            # 特殊トークンを除去
            special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<|endoftext|>', 
                            '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']
            
            for token in special_tokens:
                text = text.replace(token, '')
            
            # 改行の正規化
            text = re.sub(r'\n+', '\n', text)
            
            # 複数スペースを単一スペースに
            text = re.sub(r' +', ' ', text)
            
            # 句読点の前の不要なスペースを除去
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            
            # 文の始まりを大文字に
            sentences = text.split('.')
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    cleaned_sentences.append(sentence)
            
            text = '. '.join(cleaned_sentences)
            
            # 最後のピリオドを追加（必要に応じて）
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text
    
    def extract_text_from_multimodal_output(self, outputs: Dict[str, Any]) -> str:
        """マルチモーダル推論の出力からテキストを抽出"""
        text_candidates = []
        
        # 一般的な出力キーを優先順位付きで検索
        priority_keys = [
            'generated_text', 'text_output', 'caption', 'description',
            'answer', 'response', 'prediction', 'output_text',
            'decoded_text', 'text', 'content'
        ]
        
        # 優先キーから検索
        for key in priority_keys:
            if key in outputs:
                text_candidates.append((key, outputs[key], 10))  # 高優先度
        
        # その他のキーから検索
        for key, value in outputs.items():
            if key not in priority_keys and 'text' in key.lower():
                text_candidates.append((key, value, 5))  # 中優先度
        
        # logits系のキーを検索
        for key, value in outputs.items():
            if 'logit' in key.lower() and isinstance(value, (list, np.ndarray)):
                text_candidates.append((key, value, 1))  # 低優先度
        
        # 優先度順にソート
        text_candidates.sort(key=lambda x: x[2], reverse=True)
        
        for key, value, priority in text_candidates:
            try:
                if isinstance(value, str):
                    return self.clean_generated_text(value)
                
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        if isinstance(value[0], str):
                            return self.clean_generated_text(value[0])
                        elif isinstance(value[0], (int, float)):
                            # トークンIDとして処理
                            decoded = self.decode_token_ids(value)
                            return self.clean_generated_text(decoded)
                
                elif isinstance(value, np.ndarray):
                    if 'logit' in key.lower():
                        # ロジットとして処理
                        generated = self.process_logits_to_text(value)
                        return self.clean_generated_text(generated)
                    else:
                        # トークンIDとして処理
                        decoded = self.decode_token_ids(value)
                        return self.clean_generated_text(decoded)
                
            except Exception as e:
                logger.warning(f"Failed to process {key}: {e}")
                continue
        
        # すべて失敗した場合
        return "テキスト出力の抽出に失敗しました。"
    
    def format_conversation_output(self, text: str, 
                                 user_prompt: str = None,
                                 add_context: bool = True) -> str:
        """会話形式の出力をフォーマット"""
        try:
            if not text:
                return "応答を生成できませんでした。"
            
            # 基本クリーニング
            text = self.clean_generated_text(text)
            
            if add_context and user_prompt:
                # 文脈を考慮したフォーマット
                if "質問" in user_prompt or "?" in user_prompt or "？" in user_prompt:
                    # 質問への回答として
                    if not text.endswith(('.', '!', '?', '。', '！', '？')):
                        text += "。"
                elif "説明" in user_prompt or "教え" in user_prompt:
                    # 説明として
                    text = text.replace('\n', ' ')
                elif "生成" in user_prompt or "作成" in user_prompt:
                    # 創作として
                    pass  # そのまま
            
            return text
            
        except Exception as e:
            logger.error(f"Conversation formatting failed: {e}")
            return text if text else "フォーマットエラーが発生しました。"

# デフォルトインスタンス
default_text_output_processor = TextOutputProcessor()