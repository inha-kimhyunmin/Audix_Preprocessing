기본 라이브러리 

- numpy 배열 사용을 위한 numpy
- PyTorch 사용을 위한 torch
- PyTorch 형식으로 오디오 처리를 위한 torchaudio (mel spectrogram 변환에 사용)

## 1. sounddevice 라이브러리에서 입력이 들어오는 구조

오디오 인터페이스(여러 개의 마이크 입력을 받는 장치)에서는 아날로그 마이크 신호를 디지털 데이터로 변환한다. 이를 지속적으로 스트림으로 넘겨주는 형태

간단하게 하자면 배열에서의 인덱스 값이 주기적으로 계속해서 넘어온다.

그러면 sounddevice는 오디오 입력의 채널 수(마이크 개수), 샘플링 레이트(Hz), 샘플 길이(sampling rate * time) 에 맞춰 버퍼 할당 후

각 버퍼로 입력들이 들어온다.

[디지털 변환 과정 세부 설명](https://www.notion.so/2328f8ab094e80b198aadeb260e7300b?pvs=21)

record.py

```python
import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, SEGMENT_DURATION

def record_segment():
    """
    오디오 세그먼트를 녹음합니다.
    :return: 녹음된 오디오 데이터 (numpy 배열)
    """
    device_info = sd.query_devices(kind='input')
    channels = device_info['max_input_channels']  # type: ignore
    print(f"🎙 녹음 시작 (채널: {channels})")
    audio = sd.rec(int(SAMPLE_RATE * SEGMENT_DURATION), samplerate=SAMPLE_RATE,
                   channels=channels, dtype='float32')
    sd.wait()
    return audio  # [samples, channels]

```

여기 함수에서는 리턴값은 numpy 2차원 배열 ([[1,2,3,4,5],[6,8,4,1,2],[1,8,9,0,1]] 하나의 배열이 채널 하나 총 3개의 배열 → 3채널)

오디오 입력을 받고, for문을 돌려서 뒤의 과정은 채널별로 각각 수행

반복 횟수는 채널 개수만큼

main.py

```python
from config import NOISE_SAMPLE_PATH
from record import record_segment
from denoise import load_noise_clip, denoise
from model import load_model, separate
from mel import save_mel_tensor

def process_stream(model, repeat=5):
    """
    스트림을 처리하고 오디오를 녹음, 잡음 제거, 분리 및 저장합니다.
    
    :param model: 분리 모델
    :param repeat: 반복 횟수
    :return: None
    """
    noise_clip = load_noise_clip()
    for idx in range(repeat):
        audio = record_segment()
        for mic_idx in range(audio.shape[1]): #여기서 각 채널별로 전처리 진행
            print(f"\n🎧 마이크 {mic_idx} 처리 중")
            clean = denoise(audio[:, mic_idx], noise_clip)
            sources = separate(model, clean)
            for src_idx, src in enumerate(sources):
                save_mel_tensor(src, mic_idx, src_idx, idx)

if __name__ == "__main__":
    model = load_model()
    process_stream(model, repeat=100) # 10초 * 100회 = 1000초 (약 16분)

```

## 2. 전처리 - 1 Spectral Gating

환경 잡음 제거 - Spectral Gating  기법 사용 - noisereduce 라이브러리 사용

환경 잡음(noise sample)을 추출하기 위해 기계 소리 녹음 전 환경 소리를 녹음하는 과정이 필요하다. 이를 noise sample로 저장(.pt파일로)

### 입력 :  1차원 numpy 배열

각 채널마다의 numpy배열을 주파수 스펙트럼으로 변환(STFT 기법이 있음)

주파수 스펙트럼에서 noise sample보다 작은 소리들을 전부 제거

이를 다시 시간 영역으로 변환 

[Spectral Gating 세부 설명](https://www.notion.so/Spectral-Gating-2328f8ab094e805ab62ef4484e04eb0e?pvs=21)

### 출력 : 1차원 numpy 배열

## 3. 소리 분리 모델 - demucs

demucs 라이브러리 사용

pre-trained 모델을 사용할 때에는 그냥 demucs 라이브러리만 설치하면 되는데

따로 훈련한 모델을 사용할 때는 어떤 라이브러리를 추가로 설치해야하는지 모르겠음. 실제로 돌려봐야 알거같음

### 입력 : 2차원 numpy 배열

훈련한 모델을 불러와서 소스 분리 수행

### 출력 : tensor 파일

## 4. Mel Spectrogram 변환 → tensor 파일 변환

tensor 파일을 변환한 후 mel spectrogram으로 변환하고, 이를 tensor 파일로 변환함

tensor 파일은 종명이의 요구인 1,240,240 크기로 지정

### 입력 : tensor 파일(하나의 오디오 채널의 정보)

### 출력 : tensor 파일(.pt)

출력은 각 채널별로, 그리고 각 부품의 소리로 분리되어 .pt가 저장된다

output/
└── 2025-07-16_15-03-20/   # 측정 시간 폴더
├── mic_0/
│   ├── [fan.pt](http://fan.pt/)
│   ├── [pump.pt](http://pump.pt/)
│   └── ...
├── mic_1/
│   └── ...

output 폴더 저장 구조(ml로 넘길때는 실시간으로 넘기면 되서 상관없는데, 데이터베이스 저장 용도로 이렇게 폴더 구조 만들었음)