using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using System.Net;
using System.Net.Sockets;

// カメラ映像(JPEG)に、深度とカメラ行列を添えてPythonへUDPで送る。
// Python側はこれを使って「処理するフレーム同士の動き」を計算し、ちらつきを抑える。
//
// パケットの形式(リトルエンディアン):
//   "NRF1" | frameId(uint32) | flags(uint32, bit0=reversedZ) | near(float) | far(float)
//   | view行列(float x16, 行優先) | projection行列(float x16, 行優先)
//   | 深度の幅(uint16) | 深度の高さ(uint16) | 深度(half x 幅x高さ, 下の行から) | JPEG
[RequireComponent(typeof(Camera))]
public class ImageSender : MonoBehaviour
{
    public RenderTexture renderTexture;
    public string ipAddress = "127.0.0.1";
    public int port = 5005;
    [Tooltip("深度とカメラ行列を一緒に送る(ちらつき対策に使う)")]
    public bool sendDepth = true;
    public int depthSize = 64;

    const int MaxUdpPayload = 65507;

    private UdpClient udpClient;
    private IPEndPoint endPoint;
    private Texture2D texture2D;
    private Camera cam;
    private RenderTexture depthTexture;
    private uint frameId;

    // 最後に描画されたときのカメラ行列(RenderTextureの中身と対応させるため、描画時点の値を使う)
    private Matrix4x4 renderedView;
    private Matrix4x4 renderedProj;
    private bool hasRendered;

    void Start()
    {
        Application.targetFrameRate = 30;
        udpClient = new UdpClient();
        endPoint = new IPEndPoint(IPAddress.Parse(ipAddress), port);
        texture2D = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        cam = GetComponent<Camera>();

        if (sendDepth)
        {
            depthTexture = new RenderTexture(depthSize, depthSize, 0, RenderTextureFormat.RFloat);
            depthTexture.filterMode = FilterMode.Point;
            depthTexture.Create();
            DepthCaptureFeature.Register(cam, depthTexture);
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
        }
    }

    void OnEndCameraRendering(ScriptableRenderContext context, Camera renderedCam)
    {
        if (renderedCam != cam) return;
        renderedView = cam.worldToCameraMatrix;
        renderedProj = cam.projectionMatrix;
        hasRendered = true;
    }

    void Update()
    {
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture2D.Apply();
        RenderTexture.active = null;

        byte[] jpeg = texture2D.EncodeToJPG(75);

        if (!sendDepth || !hasRendered)
        {
            udpClient.Send(jpeg, jpeg.Length, endPoint);
            return;
        }

        var request = AsyncGPUReadback.Request(depthTexture);
        request.WaitForCompletion();
        if (request.hasError)
        {
            udpClient.Send(jpeg, jpeg.Length, endPoint);
            return;
        }
        var depth = request.GetData<float>();

        using (var ms = new MemoryStream(MaxUdpPayload))
        using (var w = new BinaryWriter(ms))
        {
            w.Write((byte)'N'); w.Write((byte)'R'); w.Write((byte)'F'); w.Write((byte)'1');
            w.Write(frameId++);
            w.Write(SystemInfo.usesReversedZBuffer ? 1u : 0u);
            w.Write(cam.nearClipPlane);
            w.Write(cam.farClipPlane);
            WriteMatrix(w, renderedView);
            WriteMatrix(w, renderedProj);
            w.Write((ushort)depthSize);
            w.Write((ushort)depthSize);
            for (int i = 0; i < depth.Length; i++)
                w.Write(Mathf.FloatToHalf(depth[i]));
            w.Write(jpeg);
            w.Flush();

            if (ms.Length > MaxUdpPayload)
            {
                Debug.LogWarning($"パケットが大きすぎます({ms.Length} bytes)。JPEGだけを送ります");
                udpClient.Send(jpeg, jpeg.Length, endPoint);
                return;
            }
            udpClient.Send(ms.GetBuffer(), (int)ms.Length, endPoint);
        }
    }

    static void WriteMatrix(BinaryWriter w, Matrix4x4 m)
    {
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
                w.Write(m[row, col]);
    }

    void OnDestroy()
    {
        if (sendDepth)
        {
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            DepthCaptureFeature.Unregister(cam);
            if (depthTexture != null) depthTexture.Release();
        }
        udpClient?.Close();
    }
}
