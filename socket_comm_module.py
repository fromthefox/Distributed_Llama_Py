"""
api of communication modules using SOCKET
"""
import socket
import threading
import struct
import struct
import pickle
import io
import torch

def send_message(sock: socket.socket, data: bytes):
    """发送带长度前缀的消息"""
    length = len(data)
    sock.sendall(struct.pack('!I', length) + data)

def receive_message(sock: socket.socket):
    """接收带长度前缀的消息"""
    try:
        # 接收4字节的长度信息
        length_data = _recv_all(sock, 4)
        if not length_data:
            return None
        
        # 解析长度
        length = struct.unpack('!I', length_data)[0]
        
        # 接收指定长度的数据
        return _recv_all(sock, length)
    except (ConnectionResetError, BrokenPipeError):
        return None
    
def _recv_all(sock: socket.socket, n: int):
    """保证接收指定长度的数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

def read_network_config(network_config: dict):
    addrs_list = network_config["addrs_list"]
    ports_list = network_config["ports_list"]
    return addrs_list, ports_list

def pack_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    tensor_bytes = buffer.getvalue()
    return tensor_bytes

def unpack_tensor(tensor_bytes):
    return torch.load(io.BytesIO(tensor_bytes))
