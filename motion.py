"""Unityから届くパケットの読み取りと、深度・カメラ行列からモーションベクトルを計算する処理。

Unity(ImageSender.cs)は、JPEGに深度とカメラ行列を添えて送ってくる。
Pythonは毎フレームは処理できず間のフレームを捨てるため、Unityの「1フレーム前からの動き」は使えない。
代わりに「前に処理したフレームのカメラ行列」と「今のフレームの深度」から、処理するフレーム同士の動きを計算する。
(静止したシーンを前提とする。動く物体の動きは含まれない)
"""
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

MAGIC = b"NRF1"
_HEADER = struct.Struct("<4sIIff16f16fHH")


@dataclass
class FrameMeta:
    frame_id: int
    reversed_z: bool
    near: float
    far: float
    view: np.ndarray  # 4x4 ワールド → カメラ
    proj: np.ndarray  # 4x4 OpenGL形式の射影行列
    depth: np.ndarray  # [h, w] 深度バッファの値(上の行から)


def parse_packet(data: bytes):
    """(JPEGのバイト列, FrameMeta または None) を返す。深度なしの古い形式にも対応"""
    if not data.startswith(MAGIC):
        return data, None
    fields = _HEADER.unpack_from(data)
    frame_id, flags, near, far = fields[1:5]
    view = np.array(fields[5:21], dtype=np.float64).reshape(4, 4)
    proj = np.array(fields[21:37], dtype=np.float64).reshape(4, 4)
    w, h = fields[37:39]
    offset = _HEADER.size
    depth = np.frombuffer(data, dtype="<f2", count=w * h, offset=offset).astype(np.float32).reshape(h, w)
    offset += w * h * 2
    # Unityのテクスチャは下の行から並んでいるので、画像と同じ上の行からに直す
    depth = depth[::-1].copy()
    meta = FrameMeta(frame_id, bool(flags & 1), near, far, view, proj, depth)
    return data[offset:], meta


def linear_eye_depth(meta: FrameMeta, raw: torch.Tensor) -> torch.Tensor:
    """深度バッファの値 → カメラからの距離"""
    n, f = meta.near, meta.far
    if meta.reversed_z:  # DirectX等: 手前が1, 奥が0
        return n * f / (n + raw * (f - n))
    return n * f / (f - raw * (f - n))


def flow_from_depth(cur: FrameMeta, prev: FrameMeta, size: int, device="cuda") -> torch.Tensor:
    """前のフレーム → 今のフレームで、各画素が画面上で何px動いたか。[2, size, size] (x右, y下が正)"""
    h, w = cur.depth.shape
    raw = torch.from_numpy(cur.depth).to(device)
    eye = linear_eye_depth(cur, raw)

    # 各画素の中心の正規化デバイス座標 (OpenGL形式: x右, y上, -1〜1)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ndc_x = (xs + 0.5) / w * 2 - 1
    ndc_y = 1 - (ys + 0.5) / h * 2

    # 今のフレームの画素 → カメラ空間 → ワールド空間 (Unityのカメラは -Z を向く)
    P = torch.from_numpy(cur.proj).float().to(device)
    view_x = (ndc_x + P[0, 2]) * eye / P[0, 0]
    view_y = (ndc_y + P[1, 2]) * eye / P[1, 1]
    view_pos = torch.stack([view_x, view_y, -eye, torch.ones_like(eye)]).reshape(4, -1)
    inv_view = torch.from_numpy(np.linalg.inv(cur.view)).float().to(device)
    world = inv_view @ view_pos

    # ワールド空間 → 前のフレームの画面上の位置
    prev_vp = torch.from_numpy(prev.proj @ prev.view).float().to(device)
    clip = prev_vp @ world
    wgt = clip[3].clamp(min=1e-4)
    prev_ndc_x = (clip[0] / wgt).reshape(h, w)
    prev_ndc_y = (clip[1] / wgt).reshape(h, w)

    flow = torch.stack([
        (ndc_x - prev_ndc_x) / 2 * size,
        -(ndc_y - prev_ndc_y) / 2 * size,
    ])
    flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    return F.interpolate(flow.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
