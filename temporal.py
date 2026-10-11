"""時間方向のちらつきを抑える処理。

どちらも「各画素が前のフレームから画面上でどれだけ動いたか」(モーションベクトル)を使う。
モーションベクトル flow は形状 [2, H, W] (画像の解像度, 単位px) で、
flow[:, y, x] = (今のフレームでの位置) - (前のフレームでの位置) とする(Unityのモーションベクトルと同じ向き)。

- NoiseWarper  : img2imgで足すノイズを景色と一緒に動かし、模様が建物に張り付くようにする
- OutputBlender: 前のフレームの出力を今の位置にずらして、今の出力と混ぜる
"""
from collections import deque

import torch
import torch.nn.functional as F


def _backward_grid(flow: torch.Tensor) -> torch.Tensor:
    """今のフレームの各画素について、前のフレームのどこから値を取ってくるかを grid_sample 用の座標で返す"""
    _, h, w = flow.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    src_x = xs - flow[0]
    src_y = ys - flow[1]
    # align_corners=False のときの正規化座標 [-1, 1]
    gx = (src_x + 0.5) / w * 2 - 1
    gy = (src_y + 0.5) / h * 2 - 1
    return torch.stack([gx, gy], dim=-1).unsqueeze(0)


def _valid_mask(grid: torch.Tensor) -> torch.Tensor:
    """前のフレームの画面内から値を取れた画素なら True"""
    return ((grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)).squeeze(0)


class NoiseWarper:
    """StreamDiffusionのノイズ(init_noise)を、モーションベクトルに沿って動かす。

    AIは画像を1/8に縮めた潜在空間のマス目で計算する。ノイズを半端な量(0.5マスなど)だけ補間してずらすと、
    隣のマスと混ざったノイズが毎フレーム変わり、かえってちらつきが増える(実測で+23%)。

    そこで各マスが「世界のどの位置を映しているか」を小数の座標で追跡し続け、
    ノイズはその座標を四捨五入したマスの値を、大きな固定のノイズ表から引く。
    動きが半端なうちはノイズは変わらず、ちょうど1マス動いたときにノイズも1マスずれるので、
    ノイズの値そのものは常に正しい標準正規分布のまま、景色に張り付く。
    """

    TABLE = 1024  # ノイズ表の一辺(マス)。座標はこの大きさで折り返す
    # 一度選んだマスは、座標がマスの中心から 0.5+HYSTERESIS 以上離れるまで切り替えない。
    # カメラのわずかな揺れで座標が境界を行き来し、ノイズが切り替わり続けるのを防ぐ
    HYSTERESIS = 0.25

    def __init__(self, stream, seed: int = 0):
        self.stream = stream
        self.scale = 8  # 画像 → 潜在空間の縮小率
        n, c, self.lh, self.lw = stream.init_noise.shape
        g = torch.Generator(device=stream.device).manual_seed(seed)
        self.table = torch.randn((n, c, self.TABLE, self.TABLE), generator=g, device=stream.device, dtype=torch.float32)
        ys, xs = torch.meshgrid(
            torch.arange(self.lh, device=stream.device, dtype=torch.float32),
            torch.arange(self.lw, device=stream.device, dtype=torch.float32),
            indexing="ij",
        )
        self.base = torch.stack([xs, ys])  # [2, lh, lw] 各マスの画面上の位置
        self.reset()

    def reset(self):
        """prepare() の後に呼ぶ。座標を初期化して履歴を消す"""
        # offset: 各マスが映している世界の位置 - 画面上の位置 (単位: マス)
        self.offset = torch.zeros_like(self.base)
        # 画面の各マスが今使っているノイズ表の位置(世界座標を整数にしたもの)。
        # 景色と一緒には動かさず、そのマスの座標が離れたときだけ選び直す
        self.cell = self.base.clone()
        # 段階kのノイズは、k-1フレーム前の位置に合わせたものを使うため履歴を持つ
        self.history = deque([self._lookup()], maxlen=self.stream.init_noise.shape[0])
        self._apply()

    def update(self, flow: torch.Tensor):
        """毎フレーム、AIに画像を渡す前に呼ぶ。flow: [2, H, W] (px)"""
        if flow is not None:
            # 画像の解像度のモーションベクトルを、潜在空間のマス単位に縮める
            flow_lat = F.avg_pool2d(flow.to(self.stream.device, torch.float32).unsqueeze(0), self.scale)[0] / self.scale
            grid = _backward_grid(flow_lat)
            # 前のフレームで対応していたマスの offset を引き継ぐ。画面外から入ってきたマスは端の値で延長する
            prev = F.grid_sample(self.offset.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
            self.offset = prev - flow_lat
        self.history.appendleft(self._lookup())
        self._apply()

    def _lookup(self) -> torch.Tensor:
        coord = self.base + self.offset
        nearest = torch.floor(coord + 0.5)
        far = (coord - self.cell).abs().amax(dim=0, keepdim=True) > 0.5 + self.HYSTERESIS
        self.cell = torch.where(far, nearest, self.cell)
        world = self.cell.long() % self.TABLE
        noise = self.table[:, :, world[1], world[0]]
        return noise.to(self.stream.init_noise.dtype)

    def _apply(self):
        # init_noise[0] と [1] は今のフレーム用、[k] (k>=2) は k-1 フレーム前のフレーム用
        noise = torch.empty_like(self.stream.init_noise)
        for k in range(noise.shape[0]):
            lag = min(max(k - 1, 0), len(self.history) - 1)
            noise[k] = self.history[lag][k]
        self.stream.init_noise = noise


class LatentFeedback:
    """前のフレームでAIが描いた結果を、モーションベクトルで今の位置にずらし、今のフレームの描き始めに混ぜる。

    StreamDiffusionは毎フレーム入力画像から描き直すため、模様がフレームごとに作り直されて流れるように変わる。
    前の結果(潜在空間)を入力の潜在表現に混ぜることで、前に描いた模様を引き継ぎつつ変わった部分だけを描き直させる。

    source_index: 引き継ぐ結果。StreamDiffusionは複数フレームを段階ごとに並行処理しており、
        前回の処理結果のうち source_index 番目は (source_index+1) フレーム前のフレームの、
        (source_index+1) 段階目まで描いた結果になる。0が最も新しく、大きいほど描き込まれているが古い。
    strength: 混ぜる割合
    update() には「引き継ぐ結果のフレーム → 今のフレーム」のモーションベクトルを渡す。
    """

    def __init__(self, stream, strength: float = 0.5, source_index: int = 0):
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

        self.stream = stream
        self.strength = strength
        self.source_index = source_index
        self.frames_back = source_index + 1
        self.prev = None
        self.flow = None
        self._retrieve_latents = retrieve_latents
        self._orig_encode = stream.encode_image
        self._orig_unet_step = stream.unet_step
        stream.encode_image = self._encode_image
        stream.unet_step = self._unet_step

    def reset(self):
        self.prev = None
        self.flow = None

    def detach(self):
        """StreamDiffusionの処理を元に戻す"""
        self.stream.encode_image = self._orig_encode
        self.stream.unet_step = self._orig_unet_step

    def update(self, flow):
        """flow: [2, H, W] (px)。引き継ぐ結果のフレームから今のフレームまでの動き。None なら混ぜない"""
        self.flow = flow

    def _unet_step(self, *args, **kwargs):
        denoised_batch, model_pred = self._orig_unet_step(*args, **kwargs)
        i = min(self.source_index, denoised_batch.shape[0] - 1)
        self.prev = denoised_batch[i : i + 1].detach().clone()
        return denoised_batch, model_pred

    def _encode_image(self, image_tensors):
        s = self.stream
        image_tensors = image_tensors.to(device=s.device, dtype=s.vae.dtype)
        img_latent = self._retrieve_latents(s.vae.encode(image_tensors), s.generator)
        img_latent = img_latent * s.vae.config.scaling_factor
        if self.prev is not None and self.flow is not None and self.strength > 0:
            scale = image_tensors.shape[-1] // img_latent.shape[-1]
            flow_lat = F.avg_pool2d(self.flow.to(img_latent.device, torch.float32).unsqueeze(0), scale)[0] / scale
            grid = _backward_grid(flow_lat)
            warped = F.grid_sample(self.prev.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            # 画面外から入ってきた部分は、前の結果がないので入力だけを使う
            m = self.strength * _valid_mask(grid).to(img_latent.dtype)
            img_latent = (1 - m) * img_latent + m * warped.to(img_latent.dtype)
        return s.add_noise(img_latent, s.init_noise[0], 0)


class OutputBlender:
    """前のフレームの出力をモーションベクトルで今の位置にずらし、今の出力と混ぜる。

    alpha は前のフレームを混ぜる割合。大きいほど安定するが、動きが速いと残像が出やすい。
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.prev = None

    def reset(self):
        self.prev = None

    def __call__(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """image: [3, H, W] (0-1), flow: [2, H, W] (px)"""
        if self.prev is None or flow is None:
            self.prev = image
            return image
        grid = _backward_grid(flow.to(image.device, torch.float32))
        warped = F.grid_sample(self.prev.unsqueeze(0).float(), grid, mode="bilinear", align_corners=False)[0]
        valid = _valid_mask(grid).to(image.dtype)
        a = self.alpha * valid
        out = (1 - a) * image + a * warped.to(image.dtype)
        self.prev = out
        return out
