import cv2
import subprocess
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# Initialize global variables to store data about detected faces and their dynamics
focus_face = 0  # Variable to potentially focus on a particular face (not used in shown code)
face_colors = {}  # Store colors for different faces, for consistent visualization
positions = {}  # Current positions of detected faces
old_positions = {}  # Previous positions of detected faces to calculate velocity
velocities = {}  # Velocities of faces calculated as the difference between positions over time
face_labels = {}  # Labels for detected faces
timestamps = []  # Timestamps of face detections

def display_face_count(frame, faces):
    """
    Adds a count of detected faces to the video frame.

    Parameters:
    - frame (np.array): The current video frame.
    - faces (list of tuples): A list of bounding boxes for each detected face.

    Returns:
    - np.array: The modified frame with the face count displayed.
    """
    # Determine the number of detected faces
    num_faces = len(faces)

    # Position to display the face count on the frame
    position = (10, 30)  # Slightly down from the top-left corner for visibility

    # Font settings for the display
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color for visibility
    line_type = 2

    # Place the text 'Faces: num_faces' on the frame
    cv2.putText(frame, f"Faces: {num_faces}", position, font, font_scale, font_color, line_type)

    return frame  # Return the frame with the face count displayed

def measure_fps(cap):
    """
    Measures the frames per second (FPS) of a video capture source.

    Parameters:
    - cap (cv2.VideoCapture): The video capture source object.

    Returns:
    - float: The measured frames per second.
    """
    # Initialize frame count and start time
    frame_count = 0
    start_time = time.time()

    # Loop to capture frames for a fixed duration (2 seconds here)
    while time.time() - start_time < 2:
        # Attempt to read a frame
        ret, _ = cap.read()
        # If frame read fails, exit loop
        if not ret:
            break
        # Increment frame count for each successful frame read
        frame_count += 1

    # Calculate elapsed time in seconds
    elapsed_time = time.time() - start_time
    # Calculate frames per second
    fps = frame_count / elapsed_time
    # Print and return the measured FPS
    print("Measured FPS: {:.2f}".format(fps))
    return fps




'''
def draw_velocity_and_position_info(frame):
    # Start vertical placement of text at y-coordinate = 50
    text_start_y = 50
    # Iterate over each center and its corresponding velocity
    for center, velocity in velocities.items():
        # Retrieve the face number associated with the current center
        face_number = face_labels[center]
        # Draw the face number on the frame
        cv2.putText(frame, f"Face {face_number}", (frame.shape[1] - 200, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the coordinates of the center of the face
        cv2.putText(frame, f"X, Y: {center}", (frame.shape[1] - 200, text_start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the horizontal velocity of the face
        cv2.putText(frame, f"X-Vel: {velocity[0]:.2f} px/s", (frame.shape[1] - 200, text_start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the vertical velocity of the face
        cv2.putText(frame, f"Y-Vel: {velocity[1]:.2f} px/s", (frame.shape[1] - 200, text_start_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Move to the next block of text 80 pixels down for the next face
        text_start_y += 80
'''

def draw_velocity_and_position_info(frame):
    text_start_y = 50
    # Iterate over each center and its corresponding velocity
    for center, velocity in velocities.items():
        # Retrieve the face number associated with the current center
        face_number = face_labels[center]
        # Draw the face number on the frame
        cv2.putText(frame, f"Face {face_number}", (frame.shape[1] - 200, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the coordinates of the center of the face
        cv2.putText(frame, f"X, Y: {center}", (frame.shape[1] - 200, text_start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the horizontal "velocity" (half of the X coordinate)
        cv2.putText(frame, f"X-Vel: {center[0] / 2:.2f} px", (frame.shape[1] - 200, text_start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255))


def draw_velocity_vectors(frame):
    # Check if there are any velocities to display
    if not velocities:
        return  # Exit if no velocities to display
    # Iterate over each center and its corresponding velocity
    for center, velocity in velocities.items():
        # Only draw a vector if there is a non-zero velocity
        if np.linalg.norm(velocity) == 0:
            continue
        # Calculate the scaling factor for the velocity vector
        scale_factor = 100 / np.linalg.norm(velocity)
        # Determine the end point of the vector
        end_point = (int(center[0] + velocity[0] * scale_factor), int(center[1] - velocity[1] * scale_factor))
        # Draw the velocity vector as an arrowed line
        cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 2)


def face_detection(frame, face_cascade):
    # Access global variables to update them
    global old_positions, velocities, face_labels, timestamps, positions
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # Check if any faces were detected
    if faces is not None and len(faces) > 0:
        # Capture the current time for timestamping this detection
        current_time = time.time()
        print("Number of faces detected:", len(faces))
        timestamps.append(current_time)
    else:
        # Print no faces detected and return
        print("No faces detected.")
        return frame, {}

    # Initialize a dictionary to store the centers of detected faces
    current_centers = {}
    # Assume a frame rate of 30 FPS for velocity calculations
    frame_time = 1 / 30

    # Process each detected face
    for idx, (x, y, w, h) in enumerate(faces):
        # Calculate the center of the face
        center = (x + w//2, y + h//2)
        # Store the center with an index
        current_centers[center] = idx
        # Choose a color for the face label
        color = (0, 0, 255) if idx == 0 else (0, 255, 0)
        # Assign a label to the face
        face_labels[center] = f"#{idx + 1}"
        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{face_labels[center]} Pos({center[0]}, {center[1]}) Vel({velocities.get(center, (0,0))[0]:.2f}, {velocities.get(center, (0,0))[1]:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Synchronize position updates
        positions[center] = (center[0], center[1])
        
        # Calculate velocity if there is a previous position recorded
        if center in old_positions:
            # Debug prints to check calculations
            print("center[0] - old_positions[center][0]", center[0] - old_positions[center][0])
            print("frame_time", frame_time)
            # Calculate velocity based on change in position and time
            velocity = ((center[0] - old_positions[center][0]) / frame_time, 
                        (center[1] - old_positions[center][1]) / frame_time)
            velocities[center] = velocity
        else:
            # Initialize velocity to zero if this is the first detection
            velocities[center] = (0, 0)
        # Update old positions for the next calculation
        old_positions[center] = center

    return frame, current_centers






def draw_bounding_boxes_and_labels(frame, faces, current_centers):
    # Loop through each detected face
    for idx, (x, y, w, h) in enumerate(faces):
        # Retrieve the center coordinates for the current face from the dictionary
        center = current_centers[idx]
        # Get the color for the current face from a dictionary or use red as default
        color = face_colors.get(center, (255, 0, 0))
        # Get the label for the current face or generate a default label using its index
        label = face_labels.get(center, f"#{idx+1}")
        # Draw a rectangle around the face using the computed coordinates and color
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Put a label above the rectangle
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def update_plot(frame_number, ax1, ax2):
    # Check if positions and timestamps lists are available and have the same length
    if positions and timestamps and len(positions) == len(timestamps):
        # Convert timestamps to datetime objects for plotting
        times = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        # Extract x and y positions for plotting
        x_positions = [pos[0] for pos in positions.values()]
        y_positions = [pos[1] for pos in positions.values()]
        # Extract x and y velocities for plotting
        x_velocities = [vel[0] for vel in velocities.values()]
        y_velocities = [vel[1] for vel in velocities.values()]

        # Print lengths of data lists to console for debugging
        print("Length of times:", len(times))
        print("Length of x_positions:", len(x_positions))
        print("Length of y_positions:", len(y_positions))

        # Clear previous data from plots
        ax1.clear()
        ax2.clear()
        
        # Plot position data on the first subplot
        ax1.plot(times, x_positions, label='X Position', color='blue')
        ax1.plot(times, y_positions, label='Y Position', color='green')
        ax1.legend()
        # Plot velocity data on the second subplot
        ax2.plot(times, x_velocities, label='X Velocity', color='red')
        ax2.plot(times, y_velocities, label='Y Velocity', color='purple')
        ax2.legend()
        
        # Set titles and labels for the axes
        ax1.set_title('Position over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Position (pixels)')
        ax2.set_title('Velocity over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Velocity (pixels/s)')

        # Format the x-axis to show time stamps
        ax1.xaxis.set_major_locator(mdates.SecondLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax2.xaxis.set_major_locator(mdates.SecondLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Rotate the date labels for better readability
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        # Print an error message if data is mismatched or insufficient
        print("Mismatch in data lengths or insufficient data.")




import cv2
import subprocess
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# Global dictionaries to store positions, velocities, and timestamps
positions = {}
timestamps = []
velocities = {}

def stream_camera():
    # Load the cascade for face detection
    cascade_path = '/home/pi4/Documents/model/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Check if the cascade was loaded successfully
    if face_cascade.empty():
        raise Exception("Failed to load cascade classifier. Check the path.")

    # Command to start video capture using libcamera-vid
    command = "libcamera-vid -t 0 --inline --listen -o tcp://0.0.0.0:8888"
    # Start the process for capturing video
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Allow some time for the video capture process to initialize
    time.sleep(2)

    # Check if the process has started successfully
    if p.poll() is not None:
        raise Exception("Failed to start libcamera-vid. Error: " + p.stderr.read().decode())

    # Connect to the video stream
    cap = cv2.VideoCapture("tcp://127.0.0.1:8888")
    # Verify the stream is opened
    if not cap.isOpened():
        raise Exception("Cannot open camera stream.")

    # Set up the plots for displaying position and velocity data
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Create an animation to update plots in real-time
    animation = FuncAnimation(fig, update_plot, fargs=(ax1, ax2), interval=1000)
    # Display the plots in a non-blocking manner
    plt.show(block=False)

    try:
        # Continuous loop to process video frames
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()
            # Check if the frame was received successfully
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Perform face detection on the frame
            frame, current_centers = face_detection(frame, face_cascade)
            # Update the frame with the count of detected faces
            frame = display_face_count(frame, list(current_centers.keys()))

            # Draw velocity vectors on the frame
            draw_velocity_vectors(frame)
            # Show the frame with annotations
            cv2.imshow('Camera Feed with Face Detection', frame)

            # Allow the user to quit the loop by pressing 'q'
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        # Terminate the subprocess
        p.terminate()
        p.wait()
        # Close the matplotlib plot
        plt.close(fig)

        # Save the position and velocity data to files if any data has been collected
        if positions and timestamps:
            # Convert timestamps to datetime objects for plotting
            times = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

            # Plot and save position data
            x_positions = [pos[0] for pos in positions.values()]
            y_positions = [pos[1] for pos in positions.values()]
            plt.figure()
            plt.plot(times, x_positions, label='X Position', color='blue')
            plt.plot(times, y_positions, label='Y Position', color='green')
            plt.title('Position over Time')
            plt.xlabel('Time')
            plt.ylabel('Position (pixels)')
            plt.legend()
            plt.gcf().autofmt_xdate()
            plt.savefig('/home/pi4/Documents/positionPlots/position_plot.png')
            plt.close()

            # Plot and save velocity data if available
            if velocities:
                x_velocities = [vel[0] for vel in velocities.values()]
                y_velocities = [vel[1] for vel in velocities.values()]
                plt.figure()
                plt.plot(times, x_velocities, label='X Velocity', color='red')
                plt.plot(times, y_velocities, label='Y Velocity', color='purple')
                plt.title('Velocity over Time')
                plt.xlabel('Time')
                plt.ylabel('Velocity (pixels/s)')
                plt.legend()
                plt.gcf().autofmt_xdate()
                plt.savefig('/home/pi4/Documents/velocityPlots/velocity_plot.png')
                plt.close()
        else:
            print("No position or velocity data available to plot.")

if __name__ == "__main__":
    stream_camera()
