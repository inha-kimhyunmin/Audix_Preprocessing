import torch

SAMPLE_RATE = 44100
MEL_SAMPLE_RATE = 16000
SEGMENT_DURATION = 10
MEL_SIZE = (240, 240)
MODEL_PATH = "model/checkpoint.th"
NOISE_SAMPLE_PATH = "noise_sample.pt"
OUTPUT_FOLDER = "output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOURCES = ["fan", "pump", "slider", "bearing", "gearbox"]  # 모델에 따라 조정
FORCE_STEREO_INPUT = True  # 모델이 2채널 입력을 요구함
CHANNELS = 2
CHANNEL_PARTS = [
    ['fan', 'pump'],              # 채널 0
    ['bearing', 'gearbox', 'slider']  # 채널 1
]