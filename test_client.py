import socket
import pandas as pd

DATA_PATH = '/home/mwilkers1/Downloads/data_exp_testFiles-1_pos_ 11.16- 3.63--4.83+rot_-0.01- 1.99- 0.04.csv'
SERVER_ADDRESS = ('127.0.0.1', 16666) # Replace with the IP address of the device running the server script

def get_dataframe(data_path):

    """Reads a CSV file and returns it as a pandas DataFrame."""

    df = pd.read_csv(data_path, sep=',', index_col='Unnamed: 0')
    return df

def transmit_data(df, filename):

    """Transmits a DataFrame as JSON over a socket connection."""

    data = filename + '|||' + df.to_json()
    data_bytes = data.encode()
    
    with socket.socket() as s:
        s.connect(SERVER_ADDRESS)
        msg = s.recv(1024)
        print(msg.decode('ascii'))
        
        total_sent = 0
        while total_sent < len(data_bytes):
            sent = s.send(data_bytes[total_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            total_sent += sent

    print(f"\tTransmitted {filename} ({total_sent} bytes).")        
    print(f"Terminated connection to server {SERVER_ADDRESS}") 

def main():

    df = get_dataframe(DATA_PATH)
    filename = 'Googyboogy-Gooooo'
    transmit_data(df, filename)

if __name__ == "__main__":
    main()
