using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.RenderGraphModule.Util;
using UnityEngine.Rendering.Universal;

// カメラの深度テクスチャを、登録された小さいRenderTextureにコピーするRenderer Feature。
// Python側で「処理するフレーム同士の動き」を計算するために使う。
// PC_Renderer の Renderer Features に追加して使う。
public class DepthCaptureFeature : ScriptableRendererFeature
{
    // 深度をコピーしたいカメラと、コピー先のRenderTexture
    static readonly Dictionary<Camera, RenderTexture> targets = new Dictionary<Camera, RenderTexture>();

    public static void Register(Camera cam, RenderTexture target) => targets[cam] = target;
    public static void Unregister(Camera cam) => targets.Remove(cam);

    class DepthCopyPass : ScriptableRenderPass
    {
        readonly Dictionary<RenderTexture, RTHandle> handles = new Dictionary<RenderTexture, RTHandle>();

        public DepthCopyPass()
        {
            renderPassEvent = RenderPassEvent.AfterRenderingTransparents;
            // 深度テクスチャを作るようURPに要求する
            ConfigureInput(ScriptableRenderPassInput.Depth);
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();
            if (!targets.TryGetValue(cameraData.camera, out var target) || target == null)
                return;

            var resourceData = frameData.Get<UniversalResourceData>();
            if (!resourceData.cameraDepthTexture.IsValid())
                return;

            if (!handles.TryGetValue(target, out var handle))
            {
                handle = RTHandles.Alloc(target);
                handles[target] = handle;
            }

            TextureHandle dst = renderGraph.ImportTexture(handle);
            // 縮小時に手前と奥の深度が混ざると存在しない距離になるため、補間しない
            renderGraph.AddBlitPass(resourceData.cameraDepthTexture, dst, Vector2.one, Vector2.zero,
                filterMode: RenderGraphUtils.BlitFilterMode.ClampNearest, passName: "Depth Capture");
        }

        public void Dispose()
        {
            foreach (var h in handles.Values)
                h.Release();
            handles.Clear();
        }
    }

    DepthCopyPass pass;

    public override void Create()
    {
        pass = new DepthCopyPass();
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        if (targets.ContainsKey(renderingData.cameraData.camera))
            renderer.EnqueuePass(pass);
    }

    protected override void Dispose(bool disposing)
    {
        pass?.Dispose();
    }
}
