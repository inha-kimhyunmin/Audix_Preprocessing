from config import NOISE_SAMPLE_PATH, CHANNEL_PARTS
from record import record_segment
from denoise import load_noise_clip, denoise
from model import load_model, separate
from mel import save_mel_tensor
from datetime import datetime
from resample import init_resampler  # 추가
import time

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
            print(f"clean : {clean}, type : {type(clean)}")
            print(f"🧹 잡음 제거 시간: {(end_denoise - start_denoise):.2f}초")

            # 2. 분리
            start_sep = time.time()
            sources = separate(model, clean)
            end_sep = time.time()
            print(f"sources = {sources}, type : {type(sources)}")
            print(f"🎛️ 소리 분리 시간: {(end_sep - start_sep):.2f}초")

            # 3. 저장
            start_save = time.time()
            parts_for_mic = CHANNEL_PARTS[mic_idx]
            for src_idx, src in enumerate(sources):
                src_name = source_names[src_idx]
                mel_result = save_mel_tensor(src, mic_idx + 1, src_name, timestamp_str, parts_for_mic)
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
