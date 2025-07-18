from demucs.htdemucs import HTDemucs
import torch
from torchaudio.transforms import Resample
from demucs.apply import apply_model
from config import MODEL_PATH, DEVICE, SAMPLE_RATE, SOURCES, FORCE_STEREO_INPUT
from monotostereo import mono_to_stereo
from resample import maybe_resample
import time

MAX_SAMPLES = 441000

def load_model():
    """
    모델을 로드하고 평가 모드로 설정합니다.
    :return: (model, sources) 튜플
    """
    print(f"📦 모델 로드 중 (Device: {DEVICE})")
    
    model = HTDemucs(sources=SOURCES)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    sources = model.sources if hasattr(model, 'sources') else [f"source_{i}" for i in range(getattr(model, 'nb_sources', 2))]
    return model, sources


def separate(model, audio_np):
    """
    오디오 데이터를 모델에 입력하여 소스 분리를 수행합니다.
    :param model: 로드된 모델
    :param audio_np: 입력 오디오 데이터 (numpy 배열)
    :return: 분리된 소스들 (torch.Tensor)
    
    """
    audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(1)  # (samples,) → (samples, 1)

    if FORCE_STEREO_INPUT:  # 현재 모델이 2채널일 경우만 복제
        audio = audio.repeat(1, 2)  # (samples, 1) → (samples, 2)

    # (samples, channels) → (batch=1, channels, samples)
    audio = audio.transpose(0, 1).unsqueeze(0).to(DEVICE)

    audio = maybe_resample(audio)

    with torch.no_grad():
        sources = apply_model(model, audio, split=True, shifts=1, progress=False)[0]
    return sources.cpu()