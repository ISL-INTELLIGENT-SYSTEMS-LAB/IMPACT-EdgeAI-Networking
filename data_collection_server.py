import socket
import pandas as pd
import os
import threading
from io import StringIO
import netifaces
import ipaddress
from datetime import datetime
import traceback
import sys

def get_server_ip():
    """
    Retrieves the local IP address of the machine running the script on the '192.168.0.0/24' network.

    Returns:
    str: The local IP address of the machine on the specific network.
    """
    # Loop through the network interfaces
    for interface in netifaces.interfaces():
        try:
            ip_info = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]
            ip_address = ip_info['addr']
            netmask = ip_info['netmask']

            # Check if the IP address is in the range of the desired network
            if ipaddress.IPv4Address(ip_address) in ipaddress.IPv4Network('192.168.0.0/24'):
                return ip_address
        except KeyError:
            continue

    return None

# Function to create a base directory for the collected data
def create_collection_dir():
    """
    This function creates a directory named 'collection' in the user's home directory where the collected data will be stored.
    
    Parameters: None
    
    Returns:
    str: The absolute path to the 'collection' directory.
    
    The function first gets the path to the user's home directory using the os.path.expanduser function. 
    It then creates a path to the 'collection' directory by joining the home directory path with the string 'collection'.
    The os.makedirs function is used to create the 'collection' directory. The exist_ok parameter is set to True, 
    which means that the function will not raise an error if the directory already exists.
    The path to the 'collection' directory is then printed to the console.
    Finally, the function returns the path to the 'collection' directory.
    """
    # Get the home directory
    home_dir = os.path.expanduser("~")
    # Create a path to the 'collection' directory
    collection_dir_path = os.path.join(home_dir, 'IMPACT_Collection')
    # Create the directory if it doesn't exist
    os.makedirs(collection_dir_path, exist_ok=True)
    # Return the path
    return collection_dir_path


# Function to create a dated directory for the current experiment
def create_experiment_directory():
    """
    This function creates a new directory for each experiment inside the 'collection' directory. 
    The name of the experiment directory is 'experiment_' followed by the current date.
    
    Parameters: None
    
    Returns:
    str: The absolute path to the newly created experiment directory.
    
    The function first calls the 'create_collection_dir' function to get the path to the 'collection' directory.
    It then gets the current date and formats it as 'YYYY-MM-DD' using the datetime.now().strftime function.
    The name of the experiment directory is created by joining 'experiment_' with the current date.
    The path to the experiment directory is created by joining the 'collection' directory path with the experiment name.
    The os.makedirs function is used to create the experiment directory. The exist_ok parameter is set to True, 
    which means that the function will not raise an error if the directory already exists.
    Finally, the function returns the path to the experiment directory.
    """
    # Create the path to the 'collection' directory
    collection_path = create_collection_dir()
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Create a directory for the current experiment inside 'collected_data'
    experiment = f'experiment_{current_date}'
    experiment_dir_path = os.path.join(collection_path, experiment)
    # Create the experiment directory if it doesn't exist
    os.makedirs(experiment_dir_path, exist_ok=True)
    # return the path
    return experiment_dir_path


# Function to process the received data
def process_data(data_bytes, client_address, total_received):
    """
    This function processes the received data from a client, converts it into a pandas DataFrame, and saves it as a CSV file.

    Parameters:
    data (bytes): The data received from the client. This is expected to be a byte string that, when decoded, 
    contains a filename and a JSON string separated by '|||'.
    client_address (tuple): The address of the client. This is a tuple containing the client's IP address and port number.
    total_received (int): The total amount of data received from the client in bytes.

    Returns: 
    tuple: The filename and the JSON data as a dictionary.

    The function first creates a directory for the current experiment using the 'create_experiment_dir' function.
    It then decodes the received data and splits it into a filename and a JSON string.
    The JSON string is converted into a pandas DataFrame using the 'pd.read_json' function.
    The DataFrame is then saved as a CSV file in the experiment directory with the filename derived from the received data.
    The function prints a message indicating the total amount of data received and the filename.
    It also prints a message indicating that the client has disconnected.
    """
    # Create a dated directory for the current experiment
    experiment_dir = create_experiment_directory()
    # Decode the data and split it into filename and JSON string
    filename, json_str = data_bytes.decode().split('|||', 1)
    # Convert the JSON string to a DataFrame
    df = pd.read_json(StringIO(json_str))
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(experiment_dir, f'{filename}.csv'), index=False)
    # Print a message indicating the total amount of data received and the filename
    print(f"Received: {total_received} bytes, '{filename}'")
    # Print a message indicating the client has disconnected
    print(f"Client {client_address} disconnected.\n")


# Function to handle a client connection
def handle_client(client_socket, client_address):
    """
    Handles a client connection by receiving data from the client and processing it.
    
    Parameters:
    client_socket (socket.socket): The socket object associated with the client. This is used to receive data from the client.
    client_address (tuple): The address of the client. This is a tuple containing the client's IP address and port number.

    Returns: None

    The function first initializes two variables: 'data_bytes' to store the received data and 'total_received' to keep track of the total amount of data received.
    It then enters a loop where it continuously receives data from the client in chunks of 1024 bytes using the 'recv' method of the client socket.
    If no data is received (i.e., 'recv' returns an empty byte string), the function breaks the loop.
    Otherwise, it adds the received data to 'data_bytes' and the length of the received data to 'total_received'.
    After all the data is received, the function calls the 'process_data' function, passing it 'data_bytes', 'client_address', and 'total_received'.
    Finally, the function closes the client socket using its 'close' method.
    """
    # Initialize the data bytes and total received
    data_bytes = b''
    total_received = 0
    # Loop until no more data is received from the client
    while True:
        # Receive data from the client
        packet = client_socket.recv(1024)
        # If no data was received, break the loop
        if not packet: 
            break
        # Add the received data to the data bytes
        data_bytes += packet
        # Add the length of the received data to the total
        total_received += len(packet)
    # Process the received data
    process_data(data_bytes, client_address, total_received)
    # Close the client socket
    client_socket.close()


# Function to start the server
def start_server():
    """
    Starts the server, listens for client connections, and starts a new thread for each client that connects.

    Parameters: None

    Returns: None

    The function first creates a server socket using the 'socket.socket' function.
    It then binds the server socket to the server address using the 'bind' method of the server socket. The server address is defined.
    The server socket starts listening for client connections with a backlog of 5 using the 'listen' method.
    A message indicating that the server is listening is printed to the console.
    The function then enters an infinite loop where it continuously accepts client connections using the 'accept' method of the server socket.
    When a client connects, the 'accept' method returns a new socket object and the address of the client.
    A message indicating that a client has connected is printed to the console, and a message is sent to the client using the 'send' method of the client socket.
    A new thread is then started to handle the client connection. The 'threading.Thread' function is used to create the new thread, 
    with the 'target' parameter set to the 'handle_client' function and the 'args' parameter set to a tuple containing the client socket and address.
    The 'start' method of the thread object is called to start the thread.
    """
    # Get the server IP address
    ip_address = get_server_ip()
    # Assign the port number
    port = 16666
    # Define the server address
    server_address = (ip_address, port) # change to correct server IPv4 address
    # Create a server socket
    server_socket = socket.socket()
    # Bind the server socket to the server address
    server_socket.bind(server_address) 
    # Start listening for client connections
    server_socket.listen(5)
    # Print the server IP address and port number
    print(f'\nServer IP: {ip_address}, Port assignment: {port}')
    # Print the path to the collection directory
    collection_dir_path = create_collection_dir()
    print(f"Collection will be stored in the '{collection_dir_path}' directory.")
    # Print a message indicating that the server is listening
    print(f'\n*** Server {server_address} is listening... ***\n')
    # Loop indefinitely
    while True:
        # Accept a client connection
        client_socket, client_address = server_socket.accept()
        # Print a message indicating a client has connected
        print(f"Client {client_address} connected.")
        # Send a message to the client indicating it has connected to the server
        client_socket.send(f'\nConnected to server {server_address}.'.encode('ascii'))
        # Start a new thread to handle the client connection
        threading.Thread(target=handle_client, args=(client_socket, client_address)).start()

# Main function
def main():
    try:
        start_server()
    except KeyboardInterrupt:
        show_traceback = input("\nKeyboardInterrupt occurred. Do you want to see the traceback? (y/n): ")
        if show_traceback.lower() == 'y':
            traceback.print_exc()

if __name__ == "__main__":
    main()
