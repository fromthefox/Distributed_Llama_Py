"""
api of communication modules using SOCKET
"""

import socket
import threading

# the socket list
connections = []

# 处理Client连接的线程
def handle_client(client_socket, addr):
    while True:
        # 等待Server发送数据到Client
        # 这里通过全局变量connections管理哪些Client需要接收数据
        # （具体的逻辑可以根据需求设计，这里仅展示发送和接收数据的流程）
        pass

# Server端主程序
def start_server(host='0.0.0.0', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server is listening on {host}:{port}")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"Accepted connection from {addr}")
            connections.append(client_socket)  # 将Client连接加入列表
            client_handler = threading.Thread(target=handle_client, args=(client_socket, addr))
            client_handler.start()

            # 后续操作可以在这里设计，例如通过选择某个Client发送数据
            # 例如：send_data_to_client(目标Client的索引, 数据)
            
    except KeyboardInterrupt:
        print("Server is shutting down.")
        for client in connections:
            client.close()
    finally:
        server_socket.close()

# 示例：向指定Client发送数据
def send_data_to_client(client_index, data):
    if 0 <= client_index < len(connections):
        client_socket = connections[client_index]
        client_socket.send(data.encode('utf-8'))
        print(f"Sent to Client {client_index}: {data}")
        response = client_socket.recv(1024).decode('utf-8')
        print(f"Received from Client {client_index}: {response}")
    else:
        print("Invalid client index.")
