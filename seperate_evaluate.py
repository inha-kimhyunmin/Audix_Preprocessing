import os
import torchaudio
import torch
import numpy as np
from model import load_model, separate  # 너의 Demucs 로딩 함수 사용
from pathlib import Path

# === 사용자 설정 ===
FOLDER = "evaluation_folder"
SOURCES = ['fan', 'pump', 'slider', 'bearing', 'gearbox']  # 오타 수정: silder -> slider

# === SI-SDR 계산 함수 ===
def compute_sisdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = torch.sum(est * ref) / torch.sum(ref ** 2)
    proj = alpha * ref
    noise = est - proj
    sisdr = 10 * torch.log10(torch.sum(proj ** 2) / torch.sum(noise ** 2))
    return sisdr.item()

# === 분리 및 평가 ===
def evaluate_demucs_on_folder(folder):
    # 1. 혼합 소리 로드
    mix_path = os.path.join(folder, "mixture.wav")
    mixture, sr = torchaudio.load(mix_path)
    if mixture.shape[0] > 1:
        mixture = mixture.mean(dim=0, keepdim=True)

    # 2. 모델 로드 및 분리
    model, _ = load_model()  # load_model()은 인자가 필요 없음
    model.eval()
    with torch.no_grad():
        estimates = separate(model, mixture.unsqueeze(0))  # shape: [1, 5, 1, time]
        estimates = estimates.squeeze(0).squeeze(1)  # shape: [5, time]

    # 3. 평가
    scores = {}
    for i, source in enumerate(SOURCES):
        gt_path = os.path.join(folder, f"{source}.wav")
        if not os.path.exists(gt_path):
            print(f"❌ {source}.wav not found, skipping...")
            continue

        target, sr2 = torchaudio.load(gt_path)
        if target.shape[0] > 1:
            target = target.mean(dim=0, keepdim=True)

        # 길이 맞추기
        est = estimates[i][:target.shape[1]].unsqueeze(0)
        target = target[:, :est.shape[1]]

        sisdr = compute_sisdr(est, target)
        scores[source] = sisdr
        print(f"✅ {source}: SI-SDR = {sisdr:.2f} dB")

    return scores

# === 실행 ===
if __name__ == "__main__":
    scores = evaluate_demucs_on_folder(FOLDER)
    print("\n📊 전체 결과:")
    for src, score in scores.items():
        print(f"{src}: {score:.2f} dB")
