import os
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from config import SAMPLE_RATE, MEL_SIZE, OUTPUT_FOLDER, MEL_SAMPLE_RATE
from datetime import datetime

def save_mel_tensor(source_tensor, mic_idx, source_name, timestamp_str, parts_to_save=None):
    """
    MelSpectrogram을 계산하고 .pt 텐서를 저장합니다.

    :param source_tensor: 입력 오디오 텐서 (shape: [1, time] 또는 [2, time])
    :param mic_idx: 마이크 인덱스
    :param source_name: 분리된 부품 이름 (예: 'fan')
    :param timestamp_str: 저장될 세그먼트 폴더명 (예: '2025-07-16_15-03-20')
    :param parts_to_save: 저장할 부품 이름 리스트 (None이면 모두 저장)

    입력 44100Hz tensor, 출력 16000Hz sampling rate mel spectrogram tensor 파일
    """
    if parts_to_save is not None and source_name not in parts_to_save:
        return
    # ✅ 멀티채널일 경우 첫 번째 채널만 선택
    if source_tensor.dim() == 2 and source_tensor.size(0) > 1:
        source_tensor = source_tensor[0].unsqueeze(0)  # [1, time]

    # 샘플링 주파수가 다르면 Resample
    if SAMPLE_RATE != MEL_SAMPLE_RATE:
        resampler = Resample(orig_freq=SAMPLE_RATE, new_freq=MEL_SAMPLE_RATE)
        source_tensor = resampler(source_tensor)

    # Mel 변환기
    mel_transform = MelSpectrogram(
        sample_rate=MEL_SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=2.0
    )
    db_transform = AmplitudeToDB(stype='power', top_db=80.0)

    with torch.no_grad():
        mel = mel_transform(source_tensor)           # [1, 128, time]
        mel = db_transform(mel)                      # dB 변환
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)  # z-score 정규화
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0), size=MEL_SIZE, mode='bilinear', align_corners=False
        ).squeeze(0)                                 # [1, 240, 240]

    mel = (mel - mel.mean()) / mel.std()
    # 크기 보정 및 저장 준비
    mel = torch.nn.functional.pad(mel, (0, max(0, MEL_SIZE[0] - mel.shape[-1])))
    mel = mel[:, :MEL_SIZE[0]]
    # 저장
    folder = os.path.join(OUTPUT_FOLDER, timestamp_str, f"mic_{mic_idx}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{source_name}.pt")
    torch.save(mel, path)
    print(f"✅ 저장 완료: {path}")
