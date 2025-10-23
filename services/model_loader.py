import os
from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

MODEL_PATH = "snumin44/simcse-ko-roberta-unsupervised"

_model = None  # 전역 캐시

def get_model():
    """필요할 때만 모델을 불러오고, 이미 있으면 재사용."""
    global _model
    if _model is None:
        print("모델 로드 시작...")
        _model = SentenceTransformer(MODEL_PATH)
        print("모델 로드 완료.")
    return _model
