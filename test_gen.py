import torchvision
import torch
from diffusers import AutoPipelineForText2Image
from streamdiffusion.image_utils import postprocess_image

print(f"--- Environment Check ---")
print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

model_id = "stabilityai/sd-turbo"
print(f"Loading model: {model_id}...")

try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to("cuda")
    print("Pipeline loaded successfully via AutoPipeline.")
except Exception as e:
    print(f"Failed to load pipeline: {e}")
    exit()

# StreamDiffusionの準備
from streamdiffusion import StreamDiffusion
stream = StreamDiffusion(pipe, t_index_list=[0], torch_dtype=torch.float16)
stream.prepare(prompt="A futuristic city in Japan, neon lights, high quality", num_inference_steps=1)

# 生成テスト（50回ループ）
print("AIがノイズから絵を構築中...")
for i in range(50):
    output_tensor = stream()
    
# 50回計算した後の結果を保存
img = postprocess_image(output_tensor, output_type="pil")[0]
img.save("output_final.png")
print("成功！ 収束した画像 'output_final.png' を確認してください。")