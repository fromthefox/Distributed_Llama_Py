import socket
import threading
from socket_comm_module import send_message, receive_message

class ClientConnection:
    """客户端连接处理器"""
    def __init__(self, sock: socket.socket, address: tuple):
        self.sock = sock
        self.address = address
        self.lock = threading.Lock()
        self.active = True

    def send_request(self, data: bytes) -> bytes:
        """线程安全的请求-响应操作"""
        with self.lock:
            if not self.active:
                raise ConnectionError("Connection closed")
            
            try:
                send_message(self.sock, data)
                return receive_message(self.sock)
            except (ConnectionError, OSError):
                self.close()
                raise

    def close(self):
        """安全关闭连接"""
        if self.active:
            self.active = False
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
            except OSError:
                pass

class ConnectionManager:
    """客户端连接管理器"""
    def __init__(self):
        self.connections = {}
        self.lock = threading.Lock()

    def add_connection(self, client_sock: socket.socket, address: tuple):
        """添加新连接"""
        with self.lock:
            self.connections[address] = ClientConnection(client_sock, address)

    def get_connection(self, address: tuple) -> ClientConnection:
        """获取指定连接"""
        with self.lock:
            return self.connections.get(address)

    def remove_connection(self, address: tuple):
        """移除失效连接"""
        with self.lock:
            if address in self.connections:
                conn = self.connections.pop(address)
                conn.close()

    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.connections.values():
                conn.close()
            self.connections.clear()

class TCPServer:
    """TCP服务端主类"""
    def __init__(self, host: str = '0.0.0.0', port: int = 9999):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.connection_manager = ConnectionManager()
        self.running = False

    def start(self):
        """启动服务监听"""
        self.running = True
        self.server_socket.listen(5)
        print(f"Server listening on {self.server_socket.getsockname()}")

        while self.running:
            try:
                client_sock, client_addr = self.server_socket.accept()
                print(f"New connection from {client_addr}")
                
                self.connection_manager.add_connection(client_sock, client_addr)
            except OSError:
                break

    def send_data(self, address: tuple, data: bytes) -> bytes:
        """向指定客户端发送数据"""
        conn = self.connection_manager.get_connection(address)
        if not conn:
            raise ConnectionError(f"No connection to {address}")
        return conn.send_request(data)

    def shutdown(self):
        """关闭服务器"""
        self.running = False
        try:
            self.server_socket.close()
        except OSError:
            pass
        self.connection_manager.close_all()

# if __name__ == "__main__":
#     import time
#     server = TCPServer(port=9999)
#     server_thread = threading.Thread(target=server.start)
#     server_thread.start()
#     time.sleep(10)
#     try:
#         # 示例使用：需要先有客户端连接
#         test_address = ('127.0.0.1', 54321)  # 假设的客户端地址
#         response = server.send_data(test_address, b"ping")
#         print(f"Received response: {response}")
#     finally:
#         server.shutdown()
#         server_thread.join()