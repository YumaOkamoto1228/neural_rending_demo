"""ちらつき(時間方向の不安定さ)を数値で測るツール。

village.png を少しずつ横にずらした画像列で「カメラの横移動」を再現し、AIに通して次の2つを測る。

- flicker : 連続する2フレームの出力を、移動量だけずらして重ねたときの平均色差(0-255)。
            入力は単なる平行移動なので、理想的な出力なら0になる。小さいほどちらつきが少ない。
- change  : 出力と入力の平均色差(0-255)。AIが元の映像をどれだけ描き変えたか。
            小さすぎるとAIの効果が薄い(元の映像のまま)ことを意味する。

移動量が分かっているので、それをモーションベクトルとして temporal.py の処理も試せる。
(実際のUnityでは奥行きによる見え方の違いや物陰があるため、ここまで理想的にはならない)

使い方:
    venv/Scripts/python.exe tools/measure_flicker.py
結果は tools/flicker_out/ に保存される(表・サンプル画像・GIF)。
"""
import os
import sys
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image
from temporal import NoiseWarper, OutputBlender
from wrapper import StreamDiffusionWrapper

MODEL_ID = "Lykon/dreamshaper-8"
# receiver.py と同じプロンプト
PROMPT = (
    "photo of a medieval european village, half-timbered houses with white plaster walls and thatched straw roofs, "
    "stone paved square, low stone wall, blue sky, natural sunlight, realistic, highly detailed"
)
NEGATIVE = "low quality, worst quality, deformed, simple, plain, cartoon, lowres"

# (t_index_list, ノイズを動かすか, 前フレームを混ぜる割合)
# t_index_listはすべて3ステップにしてTensorRTエンジンを共有する
EXPERIMENTS = [
    ([25, 40, 45], False, 0.0),  # 現在の設定
    ([30, 40, 45], False, 0.0),
    ([40, 45, 48], False, 0.0),
    ([25, 40, 45], True, 0.0),
    ([25, 40, 45], False, 0.5),
    ([25, 40, 45], True, 0.5),
    ([15, 30, 45], False, 0.0),
    ([15, 30, 45], True, 0.0),
    ([15, 30, 45], True, 0.5),
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
    """出力と入力の平均色差(出力の遅れは除去済みのものを渡す)"""
    return float(np.mean([np.abs(np.asarray(o, np.float32) - np.asarray(i, np.float32)).mean() for o, i in zip(outputs, inputs)]))


def to_tensor(img):
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().cuda() / 255


def to_pil(t):
    return Image.fromarray((t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy())


def run(wrapper, inputs, t_list, warp_noise, blend_alpha):
    wrapper.stream.t_list = t_list
    wrapper.prepare(prompt=PROMPT, negative_prompt=NEGATIVE, num_inference_steps=50, guidance_scale=1.0)
    warper = NoiseWarper(wrapper.stream) if warp_noise else None
    blender = OutputBlender(blend_alpha) if blend_alpha > 0 else None

    # カメラの横移動を表すモーションベクトル(景色は左へSHIFT動く)。全画素同じ値
    pan = torch.zeros(2, SIZE, SIZE, device="cuda")
    pan[0] = -SHIFT

    for _ in range(WARMUP):
        if warper:
            warper.update(None)
        wrapper(image=inputs[0])

    lag = len(t_list) - 1  # StreamDiffusionの出力は(ステップ数-1)フレーム遅れる
    flows = deque([None] * (lag + 1), maxlen=lag + 1)
    outputs = []
    for i, frame in enumerate(inputs):
        flow = pan if i > 0 else None
        flows.append(flow)
        if warper:
            warper.update(flow)
        out = wrapper(image=frame)
        if blender:
            # 出力は lag フレーム前の入力に対応するので、その時点のモーションベクトルを使う
            out = to_pil(blender(to_tensor(out), flows[0]))
        outputs.append(out)
    return outputs[lag:], inputs[: len(inputs) - lag]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    inputs = make_pan_frames()

    wrapper = StreamDiffusionWrapper(
        model_id_or_path=MODEL_ID,
        lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
        t_index_list=EXPERIMENTS[0][0],
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

    rows = [("input", 0.0, 0.0)]
    samples = [inputs[N_FRAMES // 2]]
    for t_list, warp_noise, alpha in EXPERIMENTS:
        outputs, matched_inputs = run(wrapper, inputs, t_list, warp_noise, alpha)
        name = f"{t_list} noise={'on' if warp_noise else 'off'} blend={alpha}"
        flick = flicker_score(outputs)
        change = change_score(outputs, matched_inputs)
        rows.append((name, flick, change))
        samples.append(outputs[N_FRAMES // 2])
        tag = f"{'_'.join(map(str, t_list))}_noise{int(warp_noise)}_blend{int(alpha * 100)}"
        small = [o.resize((256, 256)) for o in outputs]
        small[0].save(os.path.join(OUT_DIR, f"pan_{tag}.gif"), save_all=True, append_images=small[1:], duration=66, loop=0)
        print(f"{name:40s} flicker {flick:6.2f}  change {change:6.2f}  ratio {flick / change:5.2f}", flush=True)

    with open(os.path.join(OUT_DIR, "result.md"), "w", encoding="utf-8") as f:
        f.write("| 設定 | flicker (小さいほど安定) | change (描き変え量) | flicker/change |\n|---|---|---|---|\n")
        for name, flick, change in rows:
            ratio = f"{flick / change:.2f}" if change else "-"
            f.write(f"| {name} | {flick:.2f} | {change:.2f} | {ratio} |\n")

    sheet = Image.new("RGB", (SIZE * len(samples), SIZE))
    for i, img in enumerate(samples):
        sheet.paste(img, (i * SIZE, 0))
    sheet.save(os.path.join(OUT_DIR, "samples.png"))


if __name__ == "__main__":
    main()
