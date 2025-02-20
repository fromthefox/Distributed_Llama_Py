import socket
import threading
from socket_comm_module import send_message, receive_message, unpack_tensor

class TCPClient:
    """客户端实现"""
    def __init__(self, host: str = '192.168.1.104', port: int = 44444):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("192.168.1.104", 55555))
        self.sock.connect((host, port))
        self.running = True

    def start_handling(self):
        """启动请求处理循环"""
        while self.running:
            try:
                data = receive_message(self.sock)
                if not data:
                    break
                
                # 业务处理逻辑（示例：大写转换）
                data_tensor = unpack_tensor(data)
                print("data_tensor")
                print(data_tensor)
                response = b"OKT"
                
                send_message(self.sock, response)
            except (ConnectionResetError, BrokenPipeError):
                break

        self.sock.close()

if __name__ == "__main__":
    clients = []
    # for _ in range(3):  # 模拟多个客户端
    #     client = TCPClient()
    #     threading.Thread(target=client.start_handling).start()
    #     clients.append(client)
    client = TCPClient()
    client.start_handling()