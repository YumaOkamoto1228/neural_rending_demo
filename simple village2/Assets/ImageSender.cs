using UnityEngine;
using System.Net;
using System.Net.Sockets;

public class ImageSender : MonoBehaviour
{
    public RenderTexture renderTexture;
    public string ipAddress = "127.0.0.1";
    public int port = 5005;

    private UdpClient udpClient;
    private IPEndPoint endPoint;
    private Texture2D texture2D;

    void Start()
    {
        Application.targetFrameRate = 30;
        udpClient = new UdpClient();
        endPoint = new IPEndPoint(IPAddress.Parse(ipAddress), port);
        texture2D = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
    }

    void Update()
    {
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture2D.Apply();

        byte[] bytes = texture2D.EncodeToJPG(75);
        udpClient.Send(bytes, bytes.Length, endPoint);
    }
}