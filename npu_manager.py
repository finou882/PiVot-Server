"""
NPU Device Detection and OpenVINO Configuration Module
Intel NPU用のデバイス検出と設定管理
"""
import os
import logging
from typing import List, Dict, Optional
import openvino as ov
from openvino.runtime import Core

logger = logging.getLogger(__name__)

class NPUManager:
    """Intel NPU デバイスの管理とOpenVINO設定を行うクラス"""
    
    def __init__(self):
        self.core = Core()
        self.available_devices = []
        self.npu_device = None
        self._detect_devices()
        
    def _detect_devices(self) -> None:
        """利用可能なデバイスを検出"""
        try:
            self.available_devices = self.core.available_devices
            logger.info(f"Available devices: {self.available_devices}")
            
            # NPUデバイスを探す
            npu_devices = [device for device in self.available_devices 
                          if 'NPU' in device.upper()]
            
            if npu_devices:
                self.npu_device = npu_devices[0]
                logger.info(f"NPU device found: {self.npu_device}")
            else:
                logger.warning("No NPU device found. Falling back to CPU.")
                self.npu_device = "CPU"
                
        except Exception as e:
            logger.error(f"Device detection failed: {e}")
            self.npu_device = "CPU"
    
    def get_device_info(self) -> Dict[str, any]:
        """デバイス情報を取得"""
        if not self.npu_device:
            return {"error": "No device available"}
            
        try:
            device_info = {
                "selected_device": self.npu_device,
                "available_devices": self.available_devices,
                "device_properties": {}
            }
            
            # デバイスのプロパティを取得
            if self.npu_device and self.npu_device != "CPU":
                try:
                    properties = self.core.get_property(self.npu_device, "FULL_DEVICE_NAME")
                    device_info["device_properties"]["name"] = properties
                except:
                    pass
                    
            return device_info
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {"error": str(e)}
    
    def compile_model(self, model_path: str, **config) -> Optional[ov.CompiledModel]:
        """モデルをNPU用にコンパイル"""
        try:
            model = self.core.read_model(model_path)
            
            # NPU用の最適化設定
            compile_config = {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f16"  # NPU用に16bit精度を使用
            }
            compile_config.update(config)
            
            compiled_model = self.core.compile_model(
                model, 
                self.npu_device, 
                compile_config
            )
            
            logger.info(f"Model compiled successfully on {self.npu_device}")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return None
    
    def is_npu_available(self) -> bool:
        """NPUが利用可能かチェック"""
        return self.npu_device and "NPU" in self.npu_device.upper()

# Global NPU manager instance
npu_manager = NPUManager()