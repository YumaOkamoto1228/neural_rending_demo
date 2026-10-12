import socket
import select
import time
from collections import deque
import cv2
import numpy as np
import torch
import warnings
from PIL import Image
from wrapper import StreamDiffusionWrapper
from motion import parse_packet, flow_from_depth, linear_eye_depth, camera_delta
from temporal import LatentFeedback, NoiseWarper, OutputBlender

warnings.filterwarnings("ignore")
torch.manual_seed(42)

# 使用するモデル(SD1.5系・diffusers形式)。切り替えるときはここだけ変更する
# MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
# MODEL_ID = "Lykon/dreamshaper-8"

# 設定の組み合わせ。見比べるときはここだけ変更する(docs/flicker_results.md 参照)
#   "v1.0"     : ノイズ移動＋前フレーム出力の合成
#   "feedback" : v1.0 に、前フレームの結果の引き継ぎを追加(描き変えを弱めて、引き継ぎで補う)
PRESET = "feedback"

if PRESET == "feedback":
    T_INDEX_LIST = [35, 42, 47]
    FEEDBACK_STRENGTH = 0.5  # 前フレームの結果を入力に混ぜる割合
    FEEDBACK_SOURCE = 2  # 引き継ぐ結果(2 = 3フレーム前の最終結果)
else:
    T_INDEX_LIST = [25, 40, 45]
    FEEDBACK_STRENGTH = 0.0
    FEEDBACK_SOURCE = 0
SIZE = 512

# ちらつき対策(Unityから深度とカメラ行列が届いたときだけ有効)
USE_NOISE_WARP = True  # ノイズを景色と一緒に動かす
BLEND_ALPHA = 0.5  # 前のフレームの出力を混ぜる割合(0で無効)
MAX_BLEND_FLOW = 64  # これより大きく動いたフレームは混ぜない(残像を防ぐ)
FLOW_DEADZONE = 1.0  # 全画素の動きがこれ(px)未満なら静止とみなす(カメラのわずかな揺れを無視する)
DEBUG_MOTION = False  # 動きの計算が正しいか確認するための表示と画像保存(約9ms遅くなる)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 5005))
# Windowsでは、送信先(Unityの5006番)が閉じていると、送信したソケットの次の受信がエラーになる。
# Unityを止めても受信が止まらないよう、送信は別のソケットで行う
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

wrapper = StreamDiffusionWrapper(
    model_id_or_path=MODEL_ID,
    lcm_lora_id="latent-consistency/lcm-lora-sdv1-5",
    t_index_list=T_INDEX_LIST,
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

wrapper.prepare(
        # シーンの実際の見た目(白い漆喰壁・木組み・わらぶき屋根)に合わせる。ずれると建物の形まで変わってしまう
        prompt="photo of a medieval european village, half-timbered houses with white plaster walls and thatched straw roofs, "
        "stone paved square, low stone wall, blue sky, natural sunlight, realistic, highly detailed",
        #安定化のため、Negative Promptの内容を絞る
        negative_prompt="low quality, worst quality, deformed, simple, plain, cartoon, lowres",
        num_inference_steps=50,
        guidance_scale=1.0
    )

for _ in range(wrapper.batch_size):
    wrapper(image=Image.new("RGB", (SIZE, SIZE)))

warper = NoiseWarper(wrapper.stream) if USE_NOISE_WARP else None
blender = OutputBlender(BLEND_ALPHA) if BLEND_ALPHA > 0 else None
feedback = LatentFeedback(wrapper.stream, FEEDBACK_STRENGTH, FEEDBACK_SOURCE) if FEEDBACK_STRENGTH > 0 else None
# AIの出力は (ステップ数-1) フレーム遅れるので、合成にはその時点のモーションベクトルを使う
lag = len(T_INDEX_LIST) - 1
flows = deque([None] * (lag + 1), maxlen=lag + 1)
# 引き継ぐ結果は数フレーム前のものなので、そのフレームのカメラ行列から直接動きを計算する
past_metas = deque([None] * 4, maxlen=4)
prev_meta = None
flow_sum = 0.0
cam_move_sum = 0.0
cam_rot_sum = 0.0
prev_gray = None
debug_saved = False
motion_check = ""

# 処理時間の計測用（1秒ごとに平均をコンソールに表示）
stats = {"wait": 0.0, "decode": 0.0, "motion": 0.0, "ai": 0.0, "imshow": 0.0, "send": 0.0}
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

    jpeg, meta = parse_packet(data)
    img_np = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(img_rgb)
    t2 = time.perf_counter()

    # 前に処理したフレームから今のフレームまでの動き
    flow = None
    if meta is not None and prev_meta is not None:
        flow = flow_from_depth(meta, prev_meta, SIZE)
        flow_mag = flow.abs().amax().item()
        if flow_mag < FLOW_DEADZONE:
            flow = torch.zeros_like(flow)
        flow_sum += flow_mag
        cam_move, cam_rot = camera_delta(meta, prev_meta)
        cam_move_sum += cam_move
        cam_rot_sum += cam_rot
    if feedback:
        src_meta = past_metas[-feedback.frames_back]
        fb_flow = None
        if meta is not None and src_meta is not None:
            fb_flow = flow_from_depth(meta, src_meta, SIZE)
            if fb_flow.abs().amax().item() < FLOW_DEADZONE:
                fb_flow = torch.zeros_like(fb_flow)
            elif fb_flow.abs().median() > MAX_BLEND_FLOW:
                fb_flow = None  # 大きく動いたときは前の結果を引き継がない
        feedback.update(fb_flow)
    past_metas.append(meta)
    prev_meta = meta
    flows.append(flow)
    if warper:
        warper.update(flow)

    if DEBUG_MOTION and meta is not None:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if flow is not None and prev_gray is not None:
            # 画像から推定した全体のずれと、深度から計算した動き(中央値)を比べる。向きと大きさが近ければ正しい
            (sx, sy), _ = cv2.phaseCorrelate(prev_gray, gray)
            fx, fy = flow[0].median().item(), flow[1].median().item()
            if abs(sx) + abs(sy) > 2:
                motion_check = f"motion 深度から({fx:+5.1f},{fy:+5.1f}) 画像から({sx:+5.1f},{sy:+5.1f})"
        prev_gray = gray
        if not debug_saved:
            # 深度画像の上下が入力と合っているか目で確認する用(手前ほど明るい)
            eye = linear_eye_depth(meta, torch.from_numpy(meta.depth)).numpy()
            vis = np.clip(255 * (1 - np.log1p(eye) / np.log1p(100)), 0, 255).astype(np.uint8)
            cv2.imwrite("debug_depth.png", cv2.resize(vis, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
            cv2.imwrite("debug_input.png", img_np)
            debug_saved = True
    t3 = time.perf_counter()

    output_pil = wrapper(image=input_pil)

    if blender:
        out_t = torch.from_numpy(np.asarray(output_pil).copy()).cuda().permute(2, 0, 1).float() / 255
        blend_flow = flows[0]
        if blend_flow is None or blend_flow.abs().median() > MAX_BLEND_FLOW:
            blender.reset()  # 動きが分からない・大きすぎるときは混ぜない
        out_t = blender(out_t, blend_flow)
        output_rgb = (out_t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    else:
        output_rgb = np.array(output_pil)
    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    t4 = time.perf_counter()

    #比較用に新規ウィンドウで表示する処理
    cv2.imshow("AI Render View", output_bgr)
    cv2.waitKey(1)
    t5 = time.perf_counter()

    # AIで変換した画像をJPEGにエンコード
    _, encoded_img = cv2.imencode('.jpg', output_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    send_sock.sendto(encoded_img.tobytes(), ('127.0.0.1', 5006))
    t6 = time.perf_counter()

    stats["wait"] += t1 - t0
    stats["decode"] += t2 - t1
    stats["motion"] += t3 - t2
    stats["ai"] += t4 - t3
    stats["imshow"] += t5 - t4
    stats["send"] += t6 - t5
    frames += 1

    elapsed = t6 - last_report
    if elapsed >= 1.0:
        avg = {k: v / frames * 1000 for k, v in stats.items()}
        print(
            f"FPS {frames / elapsed:5.1f} | "
            f"wait {avg['wait']:5.1f}ms | decode {avg['decode']:4.1f}ms | motion {avg['motion']:4.1f}ms | "
            f"AI {avg['ai']:5.1f}ms | imshow {avg['imshow']:4.1f}ms | "
            f"send {avg['send']:4.1f}ms | dropped {dropped} | 最大の動き {flow_sum / frames:5.2f}px"
            f" | カメラ {cam_move_sum / frames * 1000:6.2f}mm {cam_rot_sum / frames:6.3f}°"
            + (" | 深度なし(JPEGのみ)" if meta is None else "")
            + (f"\n    {motion_check}" if DEBUG_MOTION and motion_check else "")
        )
        motion_check = ""
        flow_sum = 0.0
        cam_move_sum = 0.0
        cam_rot_sum = 0.0
        stats = dict.fromkeys(stats, 0.0)
        frames = 0
        dropped = 0
        last_report = t6
