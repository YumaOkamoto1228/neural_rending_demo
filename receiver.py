import socket
import select
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
    t_index_list=[25, 40, 45],
    frame_buffer_size=1,
    width=512,
    height=512,
    warmup=10,
    mode="img2img",
    use_tiny_vae=True,
    device="cuda",
    dtype=torch.float16
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

while True:
    data, addr = sock.recvfrom(65536)
    
    while select.select([sock], [], [], 0.0)[0]:
        data, addr = sock.recvfrom(65536)
    
    img_np = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(img_rgb)
    
    output_pil = wrapper(image=input_pil)
    
    output_bgr = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
    
    #比較用に新規ウィンドウで表示する処理
    cv2.imshow("AI Render View", output_bgr)
    cv2.waitKey(1)
    
    # AIで変換した画像をJPEGにエンコード
    _, encoded_img = cv2.imencode('.jpg', output_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    sock.sendto(encoded_img.tobytes(), ('127.0.0.1', 5006))