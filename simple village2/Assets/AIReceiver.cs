using System.Net.Sockets;
using System.Net;
using UnityEngine;
using UnityEngine.UI;

public class AIReceiver : MonoBehaviour
{
    public RawImage displayImage;
    private UdpClient udpClient;
    private Texture2D texture;

    void Start()
    {
        udpClient = new UdpClient(5006);
        texture = new Texture2D(512, 512);
        displayImage.texture = texture;
    }

    void Update()
    {
        if (udpClient.Available > 0)
        {
            IPEndPoint ep = new IPEndPoint(IPAddress.Any, 0);
            byte[] data = udpClient.Receive(ref ep);
            texture.LoadImage(data);
            texture.Apply();
        }
    }

    void OnDestroy()
    {
        udpClient.Close();
    }
}