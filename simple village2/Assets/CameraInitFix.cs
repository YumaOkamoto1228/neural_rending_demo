using UnityEngine;
using System.Collections;

public class CameraInitFix : MonoBehaviour
{
    private Camera cam;

    void Start()
    {
        cam = GetComponent<Camera>();
        // コルーチン（非同期処理）を開始して1フレーム待機させる
        StartCoroutine(RefreshCamera());
    }

    private IEnumerator RefreshCamera()
    {
        // カメラを一度オフにする
        cam.enabled = false;
        
        // 処理を1フレームだけ待つ（ここで他の初期化を全て終わらせる）
        yield return null;
        
        // カメラを再度オンにして、正常に描画を再開する
        cam.enabled = true;
    }
}