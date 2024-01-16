import os
import cv2
import pyzed.sl as sl
import pandas as pd
import numpy as np
import time
import math
import threading
import signal
import socket
import pickle
import nmap
import pytz
import netifaces as ni
import ipaddress
from datetime import datetime
import socket
import netifaces as ni

def get_client_ip():
    """
    Gets the client's IP address.

    Parameters: None

    Returns:
    str: The client's IP address.
    """
    # Get the name of the network interface that is currently up and has a valid IP address
    network_interface = ni.gateways()['default'][ni.AF_INET][1]
    # Get the IP address of the network interface
    ip_address = ni.ifaddresses(network_interface)[ni.AF_INET][0]['addr']
    return ip_address

# Function to get the network CIDR for a specific network interface
def get_network_cidr(interface_name='wlan0'):
    """
    Returns the network in CIDR notation for a specific network interface.
    
    Parameters:
    interface_name (str): The name of the network interface to check (default is 'wlan0').

    Returns:
    str: The network in CIDR notation, or None if the interface does not have an IPv4 address.
    """
    # Get the network addresses for the specified interface
    addresses = ni.ifaddresses(interface_name)
    # If the interface has an IPv4 address
    if ni.AF_INET in addresses:
        # Get the IPv4 address
        for link in addresses[ni.AF_INET]:
            # Get the IP address and netmask
            ip = link['addr']
            netmask = link['netmask']
            # Calculate the network CIDR
            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
            return str(network)

    return None


def get_server_address():
    """
    Retrieves the IP addresses of all devices in the same network using nmap and prompts the user to choose one.

    Parameters:
    interface_name (str): The name of the network interface to check (default is 'wlan0').

    Returns:
    tuple: The chosen IP address and port number.
    """
    network = get_network_cidr(interface_name='wlan0')
    # Scan the network for devices
    nm = nmap.PortScanner()
    # Scan the network for devices
    nm.scan(hosts=network, arguments='-sn')
    # Get the IP addresses of all devices in the network
    ips = [host for host in nm.all_hosts()]
    # Print a message before the list of IP addresses
    print(f"\n*** Found {len(ips)} devices in your network. ***\n"
        "Please select a Server IP from the following list:")
    # Prompt the user to choose an IP address
    for i, ip in enumerate(ips):
        print(f"\t{i + 1}) {ip}")
    choice = int(input("Enter a Server IP address list number: ")) - 1
    # Assign the chosen IP address
    ip_address = ips[choice]
    # Assign the port number
    port = 16666
    # Get the client's IP address and print it to the console
    client_ip = get_client_ip()
    # Print the client's IP address and the server's IP address to the console
    print(f"\nClient IP: {client_ip}, Server IP: {ip_address}")
    return (ip_address, port)


# Function to create a base directory for the collected data
def create_collection_directory():
    """
    Creates a directory named 'collection' inside the 'Documents' directory of the current user's home directory.
    If the 'collection' directory already exists, the function does not create it again.

    Parameters: None

    Returns:
    str: The absolute path to the 'collection' directory.

    The function first gets the path to the current user's home directory using the 'os.path.expanduser("~")' function and appends 'Documents' to it.
    It then appends 'collection' to this path to get the path to the 'collection' directory.
    The 'os.makedirs' function is used to create the 'collection' directory. The 'exist_ok=True' argument means that the function will not raise an error if the directory already exists.
    The function then prints a message to the console indicating where the collected data will be stored.
    Finally, the function returns the path to the 'collection' directory.
    """
    # Define the base directory path
    base_dir_path = os.path.join(os.path.expanduser("~"), "Documents")
    # Create a path to the 'collection' directory
    collection_dir_path = os.path.join(base_dir_path, 'collection')
    # Create the 'collection' directory if it doesn't exist
    os.makedirs(collection_dir_path, exist_ok=True)
    # Return the path
    return collection_dir_path


# Function to create a dated directory for the current experiment
def create_experiment_directory():
    """
    Creates a new directory for each experiment inside the 'collection' directory. 
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
    collection_path = create_collection_directory()
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Create a directory for the current experiment inside 'collected_data'
    experiment = f'experiment_{current_date}'
    experiment_dir_path = os.path.join(collection_path, experiment)
    # Create the experiment directory if it doesn't exist
    os.makedirs(experiment_dir_path, exist_ok=True)
    # return the path
    return experiment_dir_path


# Function to format the filename for the current experiment
def format_filename(trans, rot):
    """
    Formats the filename for the current experiment.

    Parameters:
    trans (tuple): A tuple containing the x, y, and z coordinates of the position.
    rot (tuple): A tuple containing the roll, pitch, and yaw angles of the rotation.

    Returns:
    str: The formatted filename.

    The function first gets the current time in the 'US/Eastern' timezone using the 'datetime.now' function and formats it as 'HH:MM:SS' using the 'strftime' method.
    It then creates the filename by joining the string 'collection_' with the current time, the position coordinates, and the rotation angles.
    The position coordinates and rotation angles are formatted as floating-point numbers with 2 decimal places.
    The position coordinates are prefixed with 'pos_' and separated by hyphens, and the rotation angles are prefixed with '+rot_' and separated by hyphens.
    Finally, the function returns the formatted filename.
    """
    # Get the current time
    timestamp = datetime.now(pytz.timezone('US/Eastern')).strftime("%H:%M:%S")
    # Return the formatted filename
    return f'collection_{timestamp}_pos_{trans[0]: .2f}-{trans[1]: .2f}-{trans[2]: .2f}+rot_{rot[0]: .2f}-{rot[1]: .2f}-{rot[2]: .2f}'


# Function to initialize the camera and set its parameters
def initialize_camera():
    """
    Initializes the ZED camera and sets its parameters.

    Parameters: None

    Returns:
    sl.Camera: The initialized ZED camera object.

    The function first creates a ZED camera object using the 'sl.Camera()' constructor.
    It then prints a message to the console indicating that object detection is running.
    The function then creates an 'InitParameters' object to set the camera parameters.
    The camera resolution is set to HD720, the camera FPS is set to 60, the coordinate units are set to feet, 
    the coordinate system is set to right-handed Y-up, and the depth mode is set to performance.
    The function then attempts to open the camera with the specified parameters using the 'open' method of the ZED camera object.
    If the 'open' method returns an error code, the function prints an error message to the console and exits the program.
    Otherwise, it prints a message to the console indicating that the camera was initialized successfully.
    Finally, the function returns the ZED camera object.
    """
    # Create a ZED camera object
    zed = sl.Camera()
    # Create a directory for the collected data and print its path to the console
    collection_dir_path = create_collection_directory()
    print(f"Collection will be stored in the '{collection_dir_path}' directory.")
    # Print a message indicating that object detection is running
    print("\nRunning object detection...")
    # Set the camera parameters
    init_params = sl.InitParameters()
    # Use HD720 video mode
    init_params.camera_resolution = sl.RESOLUTION.HD720
    # Set the camera FPS to 60  
    init_params.camera_fps = 60
    # Set the coordinate units to feet                        
    init_params.coordinate_units = sl.UNIT.FOOT
    # Set the coordinate system to right-handed Y-up
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # Set the depth mode to performance
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    # Open and close the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error initiating the ZED camera")
        print(repr(status))
        exit()
    else:
        print("ZED camera initialized successfully.")
    
    return zed


# Function to set runtime parameters for the ZED camera
def set_runtime_params():
    """
    Sets the runtime parameters for the ZED camera.

    Parameters: None

    Returns:
    sl.RuntimeParameters: The set runtime parameters for the ZED camera.

    The function first creates a 'RuntimeParameters' object using the 'sl.RuntimeParameters()' constructor.
    It then sets the confidence threshold of the runtime parameters to 50. The confidence threshold is used by the ZED camera to determine the reliability of a depth measurement.
    The function also sets the reference frame of the 3D measurements to 'WORLD'. This means that the 3D measurements will be given in the world coordinate system, which is fixed and does not change with the camera's position or orientation.
    Finally, the function returns the set runtime parameters.
    """
    # Set the runtime parameters
    runtime_params = sl.RuntimeParameters()
    # Set the confidence threshold to 50
    runtime_params.confidence_threshold = 50
    # Set the reference frame to world
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    return runtime_params


# Function to enable positional tracking on the ZED camera
def enable_positional_tracking(zed):
    """
    Enables positional tracking on the ZED camera.

    Parameters:
    zed (sl.Camera): The ZED camera object.

    Returns: None

    The function first creates a 'PositionalTrackingParameters' object using the 'sl.PositionalTrackingParameters()' constructor.
    It then sets the 'enable_imu_fusion' attribute of the positional tracking parameters to True. This enables the fusion of IMU data (accelerometer, gyroscope) with the visual (optical) odometry to improve the tracking accuracy.
    The function also sets the 'set_as_static' attribute of the positional tracking parameters to False. This means that the camera is not static and can move.
    The 'set_floor_as_origin' attribute of the positional tracking parameters is set to True. This sets the floor as the origin of the world coordinate system.
    Finally, the function enables the positional tracking on the ZED camera using the 'enable_positional_tracking' method of the ZED camera object, passing the positional tracking parameters as an argument.
    """
    # Enable positional tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # Enable IMU fusion
    positional_tracking_parameters.enable_imu_fusion = True
    # Enable positional tracking
    positional_tracking_parameters.set_as_static = False
    # Set the floor as the origin
    positional_tracking_parameters.set_floor_as_origin = True
    # Enable the positional tracking parameters    
    zed.enable_positional_tracking(positional_tracking_parameters)


# Function to enable object detection on the ZED camera
def enable_object_detection(zed):
    """
    Enables object detection on the ZED camera.

    Parameters:
    zed (sl.Camera): The ZED camera object.

    Returns: None

    The function first creates an 'ObjectDetectionParameters' object using the 'sl.ObjectDetectionParameters()' constructor.
    It then sets the detection model of the object detection parameters to 'MULTI_CLASS_BOX'. This means that the ZED camera will use the multi-class box detection model, which can detect multiple classes of objects.
    The function also sets the 'enable_tracking' attribute of the object detection parameters to True. This enables the tracking of detected objects across frames.
    Finally, the function enables object detection on the ZED camera using the 'enable_object_detection' method of the ZED camera object, passing the object detection parameters as an argument.
    """
    # Create an instance of the ObjectDetectionParameters class
    obj_param = sl.ObjectDetectionParameters()
    # Set the detection model to MULTI_CLASS_BOX
    obj_param.detection_model = sl.DETECTION_MODEL.MULTI_CLASS_BOX
    # Enable tracking
    obj_param.enable_tracking = True
    # Enable object detection on the ZED camera with the specified parameters
    zed.enable_object_detection(obj_param)


def set_object_detection_runtime_params():
    """
    Sets the object detection runtime parameters for the ZED camera.

    Parameters: None

    Returns:
    sl.ObjectDetectionRuntimeParameters: The set object detection runtime parameters for the ZED camera.

    The function first creates an 'ObjectDetectionRuntimeParameters' object using the 'sl.ObjectDetectionRuntimeParameters()' constructor.
    It then sets the detection confidence threshold of the runtime parameters to 60. The detection confidence threshold is used by the ZED camera to determine the reliability of an object detection.
    The function also sets the object class filter of the runtime parameters to only detect persons. This means that the ZED camera will only detect objects of the 'PERSON' class.
    The detection confidence threshold for the 'PERSON' class is also set to 60.
    Finally, the function returns the set object detection runtime parameters.
    """
    # Create an instance of the ObjectDetectionRuntimeParameters class
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    # Set the detection confidence threshold to 60
    detection_confidence = 60
    obj_runtime_param.detection_confidence_threshold = detection_confidence
    # Set the object class filter to only detect persons
    obj_runtime_param.object_class_filter = [sl.OBJECT_CLASS.PERSON]
    # Set the detection confidence threshold for the person class to 60
    obj_runtime_param.object_class_detection_confidence_threshold = {sl.OBJECT_CLASS.PERSON: detection_confidence} 
    # Return the object detection runtime parameters
    return obj_runtime_param


def create_sdk_output_objects(zed):
    """
    Creates the SDK output objects for the ZED camera.

    Parameters:
    zed (sl.Camera): The ZED camera object.

    Returns:
    tuple: A tuple containing the created SDK output objects.

    The function first gets the camera information from the ZED camera object using the 'get_camera_information' method.
    It then creates a 'Mat' object for the point cloud with the camera's resolution. The 'Mat' object is created with the 'sl.Mat()' constructor, passing the camera's width and height, the 'F32_C4' matrix type, and the 'CPU' memory type as arguments.
    The function also creates an 'Objects' object for object detection using the 'sl.Objects()' constructor.
    An 'Mat' object for the left camera image is created using the 'sl.Mat()' constructor with no arguments.
    A 'Resolution' object with the camera's resolution is created using the 'sl.Resolution()' constructor, passing the camera's width and height as arguments.
    A 'Pose' object for the camera's position is created using the 'sl.Pose()' constructor.
    Finally, the function returns a tuple containing the created SDK output objects.
    """
    # Get the camera information
    camera_infos = zed.get_camera_information()
    # Create a point cloud object with the camera's resolution
    point_cloud = sl.Mat(camera_infos.camera_resolution.width, camera_infos.camera_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    # Create an objects object for object detection
    objects = sl.Objects()
    # Create an image object for the left camera
    image_left = sl.Mat()
    # Create a resolution object with the camera's resolution
    display_resolution = sl.Resolution(camera_infos.camera_resolution.width, camera_infos.camera_resolution.height)
    # Create a pose object for the camera's position
    cam_w_pose = sl.Pose()
    # Return the created objects
    return point_cloud, objects, image_left, display_resolution, cam_w_pose


# Function to capture data from the ZED camera and save it to files
def capture_data(zed, dir_path, runtime_params, objects, obj_runtime_param, point_cloud, image_left, display_resolution, cam_w_pose, lock):
    """
    Captures data from the ZED camera and saves it to files.

    Parameters:
    zed (sl.Camera): The ZED camera object.
    dir_path (str): The directory path where the data will be stored.
    runtime_params (sl.RuntimeParameters): The runtime parameters for the ZED camera.
    objects (sl.Objects): The objects detected by the ZED camera.
    obj_runtime_param (sl.ObjectDetectionRuntimeParameters): The object detection runtime parameters for the ZED camera.
    point_cloud (sl.Mat): The point cloud captured by the ZED camera.
    image_left (sl.Mat): The left image captured by the ZED camera.
    display_resolution (sl.Resolution): The display resolution of the ZED camera.
    cam_w_pose (sl.Pose): The pose of the ZED camera.
    lock (threading.Lock): A lock for thread-safe operations.

    Returns:
    tuple: A tuple containing the detected objects and the filename of the saved data, or None for both if no objects were detected.

    The function first grabs the latest image from the ZED camera. If the image is successfully grabbed, it retrieves the detected objects and the camera's position.
    If objects were detected, it creates a new directory for the current experiment, gets the camera's translation and rotation, and formats the filename using the camera's translation and rotation.
    It then retrieves the point cloud and the left image from the ZED camera, and writes them to files.
    Finally, it prints the camera's location and returns the detected objects and the filename.
    If no objects were detected, it returns None for both the objects and the filename.
    """
    # Use the lock to ensure thread-safe operations
    with lock:
        # Grab the latest image from the ZED camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the detected objects from the ZED camera
            returned_state = zed.retrieve_objects(objects, obj_runtime_param)
            # Get the camera's position
            tracking_state = zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)    

            # If objects were detected
            if (returned_state == sl.ERROR_CODE.SUCCESS and objects.is_new):
                # Get the camera's translation and rotation
                trans = cam_w_pose.get_translation().get()
                rot = cam_w_pose.get_euler_angles()
                # Format the filename using the camera's translation and rotation
                filename = format_filename(trans, rot)
                # Retrieve the point cloud from the ZED camera
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, display_resolution)
                # Write the point cloud to a file
                point_cloud.write(os.path.join(dir_path, f'{filename}_pointcloud.dat'), sl.MEM.CPU) 
                # Retrieve the left image from the ZED camera
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                # Get the image data as a numpy array
                image_np = image_left.get_data()
                # Write the image to a file
                cv2.imwrite(os.path.join(dir_path, f'{filename}_image.png'), image_np)
                # Return the detected objects and the filename
                return objects, filename
    # If no objects were detected, return None for both the objects and the filename
    return None, None


# Function to process detected objects and return a DataFrame with their information
def process_objects(objects, cam_w_pose):
    """
    Processes the detected objects and returns a DataFrame with their information.

    Parameters:
    objects (sl.Objects): The objects detected by the ZED camera.
    cam_w_pose (sl.Pose): The pose of the ZED camera.

    Returns:
    pd.DataFrame: A DataFrame containing the information of the detected objects.

    The function first creates a new DataFrame with columns for the class, class confidence, label, ID, object position, object dimensions, 2D bounding box, 3D bounding box, distance from the camera, and camera position.
    It then prints the number of detected objects.
    If objects were detected, it iterates over each object, gets the object's position, calculates the straight line distance from the camera to the object, and creates a dictionary with the object's information.
    The dictionary is then added to the DataFrame.
    Finally, the function returns the DataFrame.
    If no objects were detected, it returns an empty DataFrame.
    """
    # Create a new DataFrame with the specified columns
    df = pd.DataFrame(columns=['Class','Class Confidence', 'Label', 'Id', 'Object_Position', 'Object_Dimensions', '2D_Bounding_Box', '3D_Bounding_Box', 'Distance_From_Camera', 'Camera_Position'])
    # Print the number of detected objects
    print('Number of objects detected: ', len(objects.object_list))
    # If objects were detected
    if len(objects.object_list):
        # For each detected object
        for obj in objects.object_list:
            # Get the object's position
            position = obj.position
            # Calculate the straight line distance from the camera to the object
            straight = math.sqrt(position[0]**2 + position[2]**2)
            # Create a dictionary with the object's information
            data = {'Class': obj.label, 'Class Confidence': obj.confidence,
                    'Label': obj.sublabel,
                    'Id': obj.id,
                    'Object_Position': position,
                    'Object_Dimensions': obj.dimensions,
                    '2D_Bounding_Box': obj.bounding_box_2d,
                    '3D_Bounding_Box': obj.bounding_box,
                    'Distance_From_Camera': straight,
                    'Camera_Position': cam_w_pose.pose_data()}
            # Add the dictionary to the DataFrame
            df.loc[len(df)] = data
    # Return the DataFrame
    return df


def update_camera(zed, runtime_parameters, stop, lock):
    """
    Continuously updates the ZED camera until the stop flag is set to True.

    Parameters:
    zed (sl.Camera): The ZED camera object.
    runtime_parameters (sl.RuntimeParameters): The runtime parameters for the ZED camera.
    stop (list): A list containing a single boolean value. If the value is True, the function stops updating the camera.
    lock (threading.Lock): A lock for thread-safe operations.

    Returns: None

    The function enters a while loop that continues until the stop flag is set to True.
    Inside the loop, it uses the lock to ensure thread-safe operations, and grabs the latest image from the ZED camera using the 'grab' method of the ZED camera object, passing the runtime parameters as an argument.
    """
    # Continue updating the camera until the stop flag is set to True
    while not stop[0]:
        # Use the lock to ensure thread-safe operations
        with lock:
            # Grab the latest image from the ZED camera
            zed.grab(runtime_parameters)


# Function to send the DataFrame to another device running the server script
def transmit_data(df, filename, server_address):
    """
    Transmits a DataFrame to a server.

    Parameters:
    df (pd.DataFrame): The DataFrame to be transmitted.
    filename (str): The filename to prepend to the DataFrame.

    Returns: None

    The function first defines the server address and port. It then converts the DataFrame to JSON, prepends the filename, and encodes the data as bytes.
    It creates a new socket, connects to the server, and receives a message from the server, which it prints.
    It then enters a while loop that continues until all data has been sent. Inside the loop, it sends the remaining data, checks if any data was sent, and updates the total amount of data sent.
    If no data was sent, it raises a RuntimeError.
    After all data has been sent, it prints a message indicating that the file was transmitted and that the connection was terminated.
    """
    
    # Convert the DataFrame to JSON and prepend the filename
    data = filename + '|||' + df.to_json()
    # Encode the data as bytes
    data_bytes = data.encode()
    
    # Create a new socket
    with socket.socket() as s:
        # Connect to the server
        s.connect(server_address)
        # Receive a message from the server
        msg = s.recv(1024)
        # Print the message
        print(msg.decode('ascii'))
        
        # Initialize the total amount of data sent
        total_sent = 0
        # Continue sending data until all data has been sent
        while total_sent < len(data_bytes):
            # Send the remaining data
            sent = s.send(data_bytes[total_sent:])
            # If no data was sent, raise an exception
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            # Update the total amount of data sent
            total_sent += sent
            
    # Print a message indicating that the file was transmitted
    print(f"Transmitted file: '{filename}'.")        
    # Print a message indicating that the connection was terminated
    print(f"Terminated connection to server {server_address}.\n")   


# Function to collect and process data from the ZED camera
def collect_and_process_data(zed, runtime_params, objects, obj_runtime_param, point_cloud, image_left, display_resolution, cam_w_pose, lock, stop, server_address):
    """
    Collects and processes data from the ZED camera.

    Parameters:
    zed (sl.Camera): The ZED camera object.
    runtime_params (sl.RuntimeParameters): The runtime parameters for the ZED camera.
    objects (sl.Objects): The objects detected by the ZED camera.
    obj_runtime_param (sl.ObjectDetectionRuntimeParameters): The object detection runtime parameters for the ZED camera.
    point_cloud (sl.Mat): The point cloud captured by the ZED camera.
    image_left (sl.Mat): The left image captured by the ZED camera.
    display_resolution (sl.Resolution): The display resolution of the ZED camera.
    cam_w_pose (sl.Pose): The pose of the ZED camera.
    lock (threading.Lock): A lock for thread-safe operations.
    stop (list): A list containing a single boolean value. If the value is True, the function stops collecting and processing data.

    Returns: None

    The function enters an infinite loop that continues until the stop flag is set to True.
    Inside the loop, it creates a new directory for the current experiment, waits for the user to move the camera and press Enter, and captures data from the ZED camera.
    If objects were detected and a filename was generated, it processes the detected objects, transmits the processed data to the server, and saves the processed data to a CSV file in the current experiment directory.
    It then asks the user if they want to quit or continue. If the user enters 'q' or 'Q', it sets the stop flag to True and returns from the function.
    """
    # Start an infinite loop
    while True:
        dir_path = create_experiment_directory()
        # Wait for the user to move the camera and press Enter
        input("Move the camera to a new location and press Enter to process data...\n")
        # Capture data from the ZED camera
        objects, filename = capture_data(zed, dir_path, runtime_params, objects, obj_runtime_param, point_cloud, image_left, display_resolution, cam_w_pose, lock)
        # If objects were detected and a filename was generated
        if objects and filename:
            # Process the detected objects
            df = process_objects(objects, cam_w_pose)
            # Transmit the processed data to the server
            transmit_data(df, filename, server_address)
            # Save the processed data to a CSV file in the current experiment directory
            df.to_csv(os.path.join(dir_path, f'data_exp_{filename}.csv'))

        # Ask the user if they want to quit or continue
        quit = input("Enter 'q' to quit or any other key to continue: \n")
        # If the user entered 'q' or 'Q', set the stop flag to True and return from the function
        if quit.lower() == 'q':
            stop[0] = True
            return


# Main function
def main():
    # Get the server address
    server_address = get_server_address()
    # Initialize the ZED camera
    zed = initialize_camera()
    # Set the runtime parameters for the ZED camera
    runtime_params = set_runtime_params()
    # Enable positional tracking on the ZED camera
    enable_positional_tracking(zed)
    # Enable object detection on the ZED camera
    enable_object_detection(zed)
    # Create a stop flag for the update_camera thread
    stop = [False]
    # Create a lock for thread-safe operations
    lock = threading.Lock()
    # Start a new thread that updates the camera
    threading.Thread(target=update_camera, args=(zed, runtime_params, stop, lock)).start()
    # Set the object detection runtime parameters for the ZED camera
    obj_runtime_param = set_object_detection_runtime_params()
    # Create the SDK output objects for the ZED camera
    point_cloud, objects, image_left, display_resolution, cam_w_pose = create_sdk_output_objects(zed)
    # Collect and process data from the ZED camera
    collect_and_process_data(zed, runtime_params, objects, obj_runtime_param, point_cloud, image_left, display_resolution, cam_w_pose, lock, stop, server_address)
    # Free the memory used by the image_left object
    image_left.free(sl.MEM.CPU)
    # Free the memory used by the point_cloud object
    point_cloud.free(sl.MEM.CPU)
    # Disable object detection on the ZED camera
    zed.disable_object_detection()
    # Disable positional tracking on the ZED camera
    zed.disable_positional_tracking()
    # Close the ZED camera
    zed.close()


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
