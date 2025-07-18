import os
import torchaudio
import torch
from datetime import datetime
from mel import save_mel_tensor  # 기존 함수 import
from config import SAMPLE_RATE, OUTPUT_FOLDER

# === 사용자 설정 ===
INPUT_FOLDER = "C:/Users/dotor/Desktop/2025_KEB_Project/Data/bearing_faulty"  # 부품 오디오들이 있는 폴더
MIC_IDX = 0
SOURCE_NAME = "bearing_faulty"  # 예: 'fan', 'pump', ...
TIMESTAMP_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 새로 만들 output 폴더 이름

def convert_all_single_part_audio(input_folder):
    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.wav', '.mp3', '.flac'))
    ])

    for idx, file_name in enumerate(files, 1):
        file_path = os.path.join(input_folder, file_name)

        waveform, sr = torchaudio.load(file_path)

        # 모노 처리
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 리샘플링
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # 파일 이름: fan001.pt, fan002.pt ...
        numbered_name = f"{SOURCE_NAME}{idx:03d}"

        save_mel_tensor(
            source_tensor=waveform,
            mic_idx=MIC_IDX,
            source_name=numbered_name,
            timestamp_str=TIMESTAMP_STR,
            parts_to_save=None  # 모두 저장
        )

if __name__ == "__main__":
    convert_all_single_part_audio(INPUT_FOLDER)
