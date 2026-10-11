"""ちらつき(時間方向の不安定さ)を数値で測るツール。

village.png を少しずつ横にずらした画像列で「カメラの横移動」を再現し、AIに通して次の2つを測る。

- flicker : 連続する2フレームの出力を、移動量だけずらして重ねたときの平均色差(0-255)。
            入力は単なる平行移動なので、理想的な出力なら0になる。小さいほどちらつきが少ない。
- change  : 出力と入力の平均色差(0-255)。AIが元の映像をどれだけ描き変えたか。
            小さすぎるとAIの効果が薄い(元の映像のまま)ことを意味する。

使い方:
    venv/Scripts/python.exe tools/measure_flicker.py
結果は tools/flicker_out/ に保存される(表・サンプル画像・GIF)。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image
from wrapper import StreamDiffusionWrapper

MODEL_ID = "Lykon/dreamshaper-8"
# receiver.py と同じプロンプト
PROMPT = (
    "photo of a medieval european village, half-timbered houses with white plaster walls and thatched straw roofs, "
    "stone paved square, low stone wall, blue sky, natural sunlight, realistic, highly detailed"
)
NEGATIVE = "low quality, worst quality, deformed, simple, plain, cartoon, lowres"

# 比較する t_index_list (すべて3ステップにしてTensorRTエンジンを共有する)
CANDIDATES = [
    [15, 30, 45],
    [25, 40, 45],  # 現在の設定
    [30, 40, 45],
    [35, 42, 47],
    [40, 45, 48],
]

SIZE = 512
SHIFT = 4  # 1フレームあたりの横移動量(px)
N_FRAMES = 60
WARMUP = 20
OUT_DIR = os.path.join(ROOT, "tools", "flicker_out")


def make_pan_frames():
    """village.png を高さ512に縮小し、512x512の窓をSHIFTずつ右へ動かした画像列を作る"""
    src = Image.open("village.png").convert("RGB")
    w, h = src.size
    src = src.resize((round(w * SIZE / h), SIZE), Image.LANCZOS)
    assert (N_FRAMES - 1) * SHIFT + SIZE <= src.width, "village.png の横幅が足りません"
    return [src.crop((i * SHIFT, 0, i * SHIFT + SIZE, SIZE)) for i in range(N_FRAMES)]


def flicker_score(frames):
    """連続フレームを移動量だけずらして重ねたときの平均色差"""
    diffs = []
    for a, b in zip(frames, frames[1:]):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        # カメラが右へSHIFT動く → 景色は左へSHIFT動く
        diffs.append(np.abs(a[:, SHIFT:] - b[:, :-SHIFT]).mean())
    return float(np.mean(diffs))


def change_score(outputs, inputs):
    """出力と入力の平均色差。StreamDiffusionは出力が数フレーム遅れるので、最も合う遅れ幅で比べる"""
    best = None
    for lag in range(4):
        pairs = list(zip(outputs[lag:], inputs))
        d = np.mean([np.abs(np.asarray(o, np.float32) - np.asarray(i, np.float32)).mean() for o, i in pairs])
        best = d if best is None else min(best, d)
    return float(best)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inputs = make_pan_frames()

    wrapper = StreamDiffusionWrapper(
        model_id_or_path=MODEL_ID,
        lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
        t_index_list=CANDIDATES[0],
        frame_buffer_size=1,
        width=SIZE,
        height=SIZE,
        warmup=10,
        mode="img2img",
        use_tiny_vae=True,
        device="cuda",
        dtype=torch.float16,
        acceleration="tensorrt",
    )

    rows = [("input", 0.0, flicker_score(inputs), 0.0)]
    samples = [("input", inputs[N_FRAMES // 2])]
    for t_list in CANDIDATES:
        wrapper.stream.t_list = t_list
        wrapper.prepare(prompt=PROMPT, negative_prompt=NEGATIVE, num_inference_steps=50, guidance_scale=1.0)
        for _ in range(WARMUP):
            wrapper(image=inputs[0])
        # 出力の遅れを吸収するため、先頭フレームで埋めてから流す
        outputs = [wrapper(image=f) for f in inputs]
        lag = len(t_list) - 1
        outputs = outputs[lag:]
        name = str(t_list)
        flick = flicker_score(outputs)
        change = change_score(outputs, inputs)
        rows.append((name, t_list[0], flick, change))
        samples.append((name, outputs[N_FRAMES // 2]))
        small = [o.resize((256, 256)) for o in outputs]
        small[0].save(
            os.path.join(OUT_DIR, f"pan_{'_'.join(map(str, t_list))}.gif"),
            save_all=True, append_images=small[1:], duration=66, loop=0,
        )
        print(f"{name:14s} flicker {flick:6.2f}  change {change:6.2f}", flush=True)

    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write("| t_index_list | flicker (小さいほど安定) | change (AIによる描き変え量) |\n|---|---|---|\n")
        for name, _, flick, change in rows:
            f.write(f"| {name} | {flick:.2f} | {change:.2f} |\n")

    sheet = Image.new("RGB", (SIZE * len(samples), SIZE))
    for i, (_, img) in enumerate(samples):
        sheet.paste(img, (i * SIZE, 0))
    sheet.save(os.path.join(OUT_DIR, "samples.png"))


if __name__ == "__main__":
    main()
