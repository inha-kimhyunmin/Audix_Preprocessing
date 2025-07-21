from config import NOISE_SAMPLE_PATH, CHANNEL_PARTS
from record import record_segment
from denoise import load_noise_clip, denoise
from model import load_model, separate
from mel import save_mel_tensor
from datetime import datetime
from resample import init_resampler
import time
import torch
import numpy as np

def adaptive_level_adjust(audio_np, target_rms_db=-12.0, max_gain_db=20.0, compression_threshold=0.7):
    """
    적응적 레벨 조정: 작은 소리는 증폭, 큰 소리는 압축
    
    :param audio_np: 입력 오디오 (numpy array)
    :param target_rms_db: 목표 RMS 레벨 (dB)
    :param max_gain_db: 최대 증폭 게인 (dB)
    :param compression_threshold: 압축 시작 임계값 (0~1)
    :return: 조정된 오디오 (numpy array)
    """
    if len(audio_np) == 0:
        return audio_np
    
    # 현재 RMS 계산
    current_rms = np.sqrt(np.mean(audio_np ** 2))
    
    if current_rms < 1e-8:  # 무음에 가까운 경우
        print(f"⚠️ Nearly silent audio (RMS: {current_rms:.8f}), skipping adjustment")
        return audio_np
    
    # 목표 RMS 레벨
    target_rms = 10 ** (target_rms_db / 20.0)
    current_rms_db = 20 * np.log10(current_rms + 1e-8)
    
    # 최대값 확인 (클리핑 방지용)
    max_val = np.max(np.abs(audio_np))
    
    if current_rms < target_rms:
        # 🔊 작은 소리: 증폭
        gain_factor = target_rms / current_rms
        
        # 최대 증폭 제한
        max_gain_factor = 10 ** (max_gain_db / 20.0)
        gain_factor = min(gain_factor, max_gain_factor)
        
        adjusted_audio = audio_np * gain_factor
        
        # 클리핑 방지
        new_max = np.max(np.abs(adjusted_audio))
        if new_max > 0.95:
            adjusted_audio = adjusted_audio * (0.95 / new_max)
            actual_gain = 20 * np.log10((0.95 / new_max) * gain_factor)
        else:
            actual_gain = 20 * np.log10(gain_factor)
            
        print(f"🔊 Amplified: {current_rms_db:.1f}dB → {target_rms_db:.1f}dB (+{actual_gain:.1f}dB)")
        
    elif max_val > compression_threshold:
        # 🔇 큰 소리: 소프트 압축 (리미팅)
        # Soft knee compression
        ratio = 3.0  # 압축 비율
        threshold = compression_threshold
        
        # 압축 적용
        adjusted_audio = np.copy(audio_np)
        over_threshold = np.abs(adjusted_audio) > threshold
        
        # 압축 함수: 임계값을 넘는 부분을 소프트하게 압축
        over_amount = np.abs(adjusted_audio[over_threshold]) - threshold
        compressed_over = threshold + over_amount / ratio
        
        # 원래 부호 유지
        adjusted_audio[over_threshold] = np.sign(adjusted_audio[over_threshold]) * compressed_over
        
        # RMS를 목표 레벨로 조정
        new_rms = np.sqrt(np.mean(adjusted_audio ** 2))
        if new_rms > 1e-8:
            rms_adjust = target_rms / new_rms
            adjusted_audio = adjusted_audio * rms_adjust
        
        final_rms_db = 20 * np.log10(np.sqrt(np.mean(adjusted_audio ** 2)) + 1e-8)
        print(f"🔇 Compressed & normalized: {current_rms_db:.1f}dB → {final_rms_db:.1f}dB")
        
    else:
        # 📊 적절한 범위: 약간의 조정만
        gain_factor = target_rms / current_rms
        # 제한된 조정 (±3dB)
        gain_factor = np.clip(gain_factor, 10**(-3/20), 10**(3/20))
        
        adjusted_audio = audio_np * gain_factor
        final_rms_db = 20 * np.log10(np.sqrt(np.mean(adjusted_audio ** 2)) + 1e-8)
        print(f"📊 Minor adjustment: {current_rms_db:.1f}dB → {final_rms_db:.1f}dB")
    
    return adjusted_audio

def process_stream(model, source_names, repeat=5):
    """
    오디오 스트림을 처리하고 저장합니다.
    
    :param model: 분리 모델
    :param source_names: 부품 이름 리스트 (예: ['fan', 'pump', ...])
    :param repeat: 반복 횟수
    """
    noise_clip = load_noise_clip()

    for i in range(repeat):
        print(f"\n📡 반복 {i+1}/{repeat}")

        start_record = time.time()
        audio = record_segment()
        end_record = time.time()
        print(f"🎙️ 녹음 완료: {(end_record - start_record):.2f}초")
        print("audio.shape = ", audio.shape)
        print(f"audio = {audio}, type : {type(audio)}")

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for mic_idx in range(audio.shape[1]):
            print(f"\n🎧 마이크 {mic_idx + 1} 처리 중")
            start_total = time.time()

            # 1. 잡음 제거
            start_denoise = time.time()
            clean = denoise(audio[:, mic_idx], noise_clip)
            end_denoise = time.time()
            print(f"🧹 잡음 제거 시간: {(end_denoise - start_denoise):.2f}초")
            
            # 1.5. 적응적 레벨 조정 (RMS를 -18dB 목표로 - 기계음에 적합)
            # 조용한 환경의 미세한 기계음: -15dB
            # 시끄러운 공장 환경: -21dB  
            # 일반적인 산업 환경: -18dB (현재 설정)
            start_adjust = time.time()
            clean = adaptive_level_adjust(clean, target_rms_db=-18.0)
            end_adjust = time.time()
            print(f"⚖️ 레벨 조정 시간: {(end_adjust - start_adjust):.2f}초")

            # 2. 분리
            start_sep = time.time()
            sources = separate(model, clean)
            end_sep = time.time()
            print(f"🎛️ 소리 분리 시간: {(end_sep - start_sep):.2f}초")

            # 3. 저장
            start_save = time.time()
            parts_for_mic = CHANNEL_PARTS[mic_idx]
            for src_idx, src in enumerate(sources):
                src_name = source_names[src_idx]
                mel_result = save_mel_tensor(src, mic_idx, src_name, timestamp_str, parts_for_mic)
                print(f"📁 저장 완료: {src_name}")
            end_save = time.time()
            print(f"💾 저장 시간: {(end_save - start_save):.2f}초")

            # 총 시간
            end_total = time.time()
            print(f"⏱️ 마이크 {mic_idx + 1} 전체 처리 시간: {(end_total - start_total):.2f}초")

if __name__ == "__main__":
    model, source_names = load_model()
    init_resampler(model.samplerate)
    process_stream(model, source_names, repeat=100)
