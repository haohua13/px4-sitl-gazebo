import socket
import struct

# to send visual measurement information to the simulink model
def send_message(W, q):
    # IP address and port of the MATLAB receiver
    matlab_ip = "127.0.0.1"
    matlab_port = 4337
    # Create a UDP socket
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Create the message
    list = [W[0], W[1], W[2], q[0], q[1], q[2]]
    message = struct.pack('%sd' % len(list), *list)
    # Send the message to MATLAB
    sender_socket.sendto(message, (matlab_ip, matlab_port))
    print("sent message:", message)
    # Close the socket
    sender_socket.close()


# to receive angular velocity and rotation from the simulink model
def receive_message():
    udp_ip= "0.0.0.0"
    udp_port = 6337
    sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
    sock.bind((udp_ip, udp_port))
    msgcount = 0
    data, addr = sock.recvfrom(512)
    # print("len", str(len(data)))
    output = struct.unpack('dddddd', data)
    msgcount += 1
    print("received message:", output)
    # print("message count:", str(msgcount))
    return output
