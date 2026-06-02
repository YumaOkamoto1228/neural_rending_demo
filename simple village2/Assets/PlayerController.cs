using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float speed = 5.0f;
    public float sensitivity = 2.0f;
    public float jumpForce = 5.0f; // ジャンプ力の追加
    public Transform playerCamera;
    
    private float xRotation = 0.0f;
    private Rigidbody rb; // Rigidbodyの追加

    void Start()
    {
        // アタッチされているRigidbodyを取得
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        float moveX = Input.GetAxis("Horizontal");
        float moveZ = Input.GetAxis("Vertical");
        transform.position += (transform.right * moveX + transform.forward * moveZ) * speed * Time.deltaTime;

        float mouseX = Input.GetAxis("Mouse X") * sensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * sensitivity;

        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);

        playerCamera.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
        transform.Rotate(Vector3.up * mouseX);

        // スペースキーが押されたらジャンプ
        if (Input.GetKeyDown(KeyCode.Space))
        {
            rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
        }
    }
}