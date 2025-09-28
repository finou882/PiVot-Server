# PiVot-Server Configuration File
# サーバー設定

# Server Settings
HOST = "0.0.0.0"
PORT = 8000
RELOAD = False
LOG_LEVEL = "info"

# NPU Settings
NPU_PERFORMANCE_HINT = "LATENCY"  # LATENCY, THROUGHPUT
NPU_INFERENCE_PRECISION = "f16"    # f16, f32

# Model Settings  
DEFAULT_MODEL_TYPE = "vision"
MAX_LOADED_MODELS = 5

# Image Processing Settings
DEFAULT_IMAGE_SIZE = (224, 224)
MAINTAIN_ASPECT_RATIO = False
IMAGE_NORMALIZE = True

# ImageNet標準化パラメータ
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Text Processing Settings  
DEFAULT_MAX_TEXT_LENGTH = 512
DEFAULT_TOKENIZER = "bert-base-uncased"
USE_FAST_TOKENIZER = True

# API Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
CORS_ALLOW_ORIGINS = ["*"]

# Benchmark Settings
DEFAULT_BENCHMARK_ITERATIONS = 10
BENCHMARK_WARMUP_ITERATIONS = 3

# Logging Settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = None  # None for console only, or specify file path

# Model Paths (examples)
MODEL_DIRECTORY = "./models"
SAMPLE_MODELS = {
    "resnet50": {
        "path": "./models/resnet50.xml",
        "type": "vision", 
        "description": "ResNet-50 image classification"
    },
    "bert_base": {
        "path": "./models/bert_base.xml",
        "type": "text",
        "description": "BERT base text processing"
    },
    "clip": {
        "path": "./models/clip.xml", 
        "type": "multimodal",
        "description": "CLIP vision-language model"
    }
}