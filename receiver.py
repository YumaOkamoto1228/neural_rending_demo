import socket
import select
import time
import cv2
import numpy as np
import torch
import warnings
from PIL import Image
from wrapper import StreamDiffusionWrapper

warnings.filterwarnings("ignore")
torch.manual_seed(42)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 5005))

wrapper = StreamDiffusionWrapper(
    model_id_or_path="runwayml/stable-diffusion-v1-5",
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
    t_index_list=[32, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    mode="img2img",
    use_tiny_vae=True,
    device="cuda",
    dtype=torch.float16,
    acceleration="tensorrt",
)

wrapper.prepare(
        prompt="photorealistic, masterpiece, highly detailed medieval village, stone wall, orange brick house, Black roof, dirt and cobblestone path, 4k reso",
        #安定化のため、Negative Promptの内容を絞る
        negative_prompt="low quality, worst quality, deformed, simple, plain, cartoon, lowres",
        num_inference_steps=50,
        guidance_scale=1.0 
    )

for _ in range(wrapper.batch_size):
    wrapper(image=Image.new("RGB", (512, 512)))

# 処理時間の計測用（1秒ごとに平均をコンソールに表示）
stats = {"wait": 0.0, "decode": 0.0, "ai": 0.0, "imshow": 0.0, "send": 0.0}
frames = 0
dropped = 0
last_report = time.perf_counter()

while True:
    t0 = time.perf_counter()
    data, addr = sock.recvfrom(65536)
    t1 = time.perf_counter()

    # 溜まっている古いフレームを捨てて、最新の1枚だけを使う
    while select.select([sock], [], [], 0.0)[0]:
        data, addr = sock.recvfrom(65536)
        dropped += 1

    img_np = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(img_rgb)
    t2 = time.perf_counter()

    output_pil = wrapper(image=input_pil)

    output_bgr = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    t3 = time.perf_counter()

    #比較用に新規ウィンドウで表示する処理
    cv2.imshow("AI Render View", output_bgr)
    cv2.waitKey(1)
    t4 = time.perf_counter()

    # AIで変換した画像をJPEGにエンコード
    _, encoded_img = cv2.imencode('.jpg', output_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    sock.sendto(encoded_img.tobytes(), ('127.0.0.1', 5006))
    t5 = time.perf_counter()

    stats["wait"] += t1 - t0
    stats["decode"] += t2 - t1
    stats["ai"] += t3 - t2
    stats["imshow"] += t4 - t3
    stats["send"] += t5 - t4
    frames += 1

    elapsed = t5 - last_report
    if elapsed >= 1.0:
        avg = {k: v / frames * 1000 for k, v in stats.items()}
        print(
            f"FPS {frames / elapsed:5.1f} | "
            f"wait {avg['wait']:5.1f}ms | decode {avg['decode']:4.1f}ms | "
            f"AI {avg['ai']:5.1f}ms | imshow {avg['imshow']:4.1f}ms | "
            f"send {avg['send']:4.1f}ms | dropped {dropped}"
        )
        stats = dict.fromkeys(stats, 0.0)
        frames = 0
        dropped = 0
        last_report = t5
