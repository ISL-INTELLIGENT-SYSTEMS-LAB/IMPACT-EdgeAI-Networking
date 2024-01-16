# Environment requirements
- python3.8
- pip 23.2.1
- mvenv (to create, activate, and deactivate the virtual environment)
- python3.8-dev (obtained by sudo apt-get install python3.8-dev)
- CUDA 10.2

## data_collection_client.py requirements
- Cython
- scikit-build
- numpy
- pandas
- opencv-contrib-python
- pyzed (ZED Python API, run get_python_api.py located in /usr/local/zed)

## network_relation.py requirements
- matplotlib

## utils.py requirements
- torch->dependency for pytorch3d
- torchvision->dependency for pytorch3d
- torchaudio->dependency for pytorch3d
- fvcore->dependency for pytorch3d
- iopath->dependency for pytorch3d
- pytorch3d (pip install "git+https://github.com/facebookresearch/pytorch3d.git")
- scipy

## Data Collection Client Explained
This Python script is designed to control a ZED camera, capture data, process it, and send it to a server. Here are the main ideas:

Initialization: The script starts by setting up directories for data storage based on the current date and time. It also defines a function to format filenames based on camera translation and rotation values.

Camera Setup: The script initializes the ZED camera and sets its parameters, including resolution, FPS, coordinate units, and depth mode. It also enables positional tracking and object detection on the camera.

Data Capture: The script captures data from the camera, including point cloud data and images. It saves this data to files with a specific filename format. The capture process is done in a separate thread to allow continuous data capture.

Data Processing: The script processes the captured data, specifically the detected objects. It extracts information about each object, such as its class, confidence, position, dimensions, and distance from the camera. This information is stored in a pandas DataFrame.

Data Transmission: The script sends the DataFrame to a server using a socket connection. It encodes the DataFrame as a JSON string and sends it over the socket. If the data is not fully sent in one go, it continues to send the remaining data until all data is sent.

Main Loop: The main function of the script runs a loop that prompts the user to move the camera and press Enter to process data. It captures data, processes it, sends it to the server, and saves it to a CSV file. The loop continues until the user decides to quit.

Cleanup: After the main loop ends, the script frees up the memory used by the image and point cloud data, disables object detection and positional tracking, and closes the camera.

## Data Collection Server Explained

The main ideas behind this code are:

Server Setup: The script sets up a server that listens for client connections on a specified address and port.

Multithreading: For each client that connects, the script starts a new thread to handle the connection. This allows the server to handle multiple clients simultaneously.

Data Reception: The script receives data from the client in chunks and concatenates them into a byte string.

Data Processing: The script processes the received data by decoding it, splitting it into a filename and a JSON string, converting the JSON string into a pandas DataFrame, and saving the DataFrame to a CSV file.

Directory Management: If the directory where the CSV files are to be saved does not exist, the script creates it.
