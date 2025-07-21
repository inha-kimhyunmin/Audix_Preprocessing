import os
import torchaudio
import torch
import random
from datetime import datetime
from mel import save_mel_tensor
from config import SAMPLE_RATE, OUTPUT_FOLDER

# === 사용자 설정 ===
INPUT_FOLDER = "D:/machine_sounds"
MIC_IDX = 0
TIMESTAMP_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# === 증폭 설정 ===
TARGET_RMS_DB = -12.0  # 고정된 목표 레벨 (일관성을 위해)
# TARGET_RMS_DB_RANGE = (-12.0, -6.0)  # 랜덤 범위 (주석 처리)

def amplify_audio(waveform):
    """
    오디오를 고정된 목표 레벨로 증폭합니다 (일관성을 위해).
    
    :param waveform: 입력 오디오 텐서 (torch.Tensor)
    :return: 증폭된 오디오 텐서
    """
    # 현재 RMS 계산
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    
    if current_rms < 1e-8:  # 무음에 가까운 경우
        print(f"⚠️ Nearly silent audio (RMS: {current_rms:.8f}), skipping amplification")
        return waveform
    
    # 고정된 목표 RMS 레벨 사용 (CNN 학습을 위한 일관성)
    target_rms = 10 ** (TARGET_RMS_DB / 20.0)
    
    # 증폭 팩터 계산
    amplification_factor = target_rms / current_rms
    
    # 증폭 적용
    amplified_audio = waveform * amplification_factor
    
    # 클리핑 체크 및 조정
    max_val = torch.max(torch.abs(amplified_audio))
    if max_val > 0.95:
        # 클리핑 방지를 위해 스케일링
        safe_factor = 0.95 / max_val
        amplified_audio = amplified_audio * safe_factor
        actual_amplification = amplification_factor * safe_factor
        print(f"🔧 Clipping prevented, reduced amplification")
    else:
        actual_amplification = amplification_factor
    
    # 최종 RMS 확인
    final_rms = torch.sqrt(torch.mean(amplified_audio ** 2))
    final_rms_db = 20 * torch.log10(final_rms + 1e-8)  # 텐서로 유지
    
    # 간헐적으로 증폭 정보 출력
    if random.random() < 0.1:  # 10% 확률로 출력
        gain_db = 20 * torch.log10(torch.tensor(actual_amplification) + 1e-8)  # 텐서로 변환
        print(f"🔊 Amplification: {current_rms:.6f} → {final_rms:.6f} "
              f"(+{gain_db.item():.1f}dB, target: {TARGET_RMS_DB:.1f}dB, final: {final_rms_db.item():.1f}dB)")
    
    return amplified_audio

def convert_all_audio_recursively(input_folder):
    folder_counts = {}  # 각 폴더 이름별 카운터

    for root, _, files in os.walk(input_folder):
        files = sorted([f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac'))])
        if not files:
            continue

        # 폴더 이름 얻기
        folder_name = os.path.basename(root)
        if folder_name == os.path.basename(input_folder):
            # 최상위 폴더면 스킵 (파일이 있을 경우에만 해당됨)
            continue

        folder_counts.setdefault(folder_name, 1)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                waveform, sr = torchaudio.load(file_path)

                # 모노 처리
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # 리샘플링
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    waveform = resampler(waveform)

                # 🔊 오디오 증폭 (-12dB ~ -6dB 범위로)
                waveform = amplify_audio(waveform)

                # 파일 이름: foldername001.pt, foldername002.pt, ...
                count = folder_counts[folder_name]
                numbered_name = f"{folder_name}{count:03d}"
                folder_counts[folder_name] += 1

                save_mel_tensor(
                    source_tensor=waveform,
                    mic_idx=MIC_IDX,
                    source_name=numbered_name,
                    timestamp_str=TIMESTAMP_STR,
                    parts_to_save=None
                )

                print(f"✅ Processed: {file_path} → {numbered_name}")

            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")

if __name__ == "__main__":
    convert_all_audio_recursively(INPUT_FOLDER)
