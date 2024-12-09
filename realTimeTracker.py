import cv2
import subprocess
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize global variables
focus_face = 0
face_colors = {}
positions = {}
old_positions = {}
velocities = {}
face_labels = {}


def display_face_count(frame, faces):
    """
    Display the number of detected faces on the frame.

    Parameters:
    - frame: The current video frame
    - faces: A list of bounding boxes for each detected face

    Returns:
    - frame: The modified frame with the face count displayed
    """
    # Number of faces
    num_faces = len(faces)

    # Position for the text
    position = (10, 30)  # Slightly down from the top-left corner to be visible

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color
    line_type = 2

    # Put text on the frame
    cv2.putText(frame, f"Faces: {num_faces}", position, font, font_scale, font_color, line_type)

    return frame


def measure_fps(cap):
    # Measure the frames per second of the video capture
    frame_count = 0
    start_time = time.time()
    while time.time() - start_time < 2:  # Measure for 2 seconds
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print("Measured FPS: {:.2f}".format(fps))
    return fps


def draw_velocity_and_position_info(frame):
    text_start_y = 50
    for center, velocity in velocities.items():
        face_number = face_labels[center]
        cv2.putText(frame, f"Face {face_number}", (frame.shape[1] - 200, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"X, Y: {center}", (frame.shape[1] - 200, text_start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f"X-Vel: {velocity[0]:.2f} px/s", (frame.shape[1] - 200, text_start_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Y-Vel: {velocity[1]:.2f} px/s", (frame.shape[1] - 200, text_start_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        text_start_y += 80  # Adjust spacing for next face info



def draw_velocity_vectors(frame):
    if not velocities:
        return
    for center, velocity in velocities.items():
        if np.linalg.norm(velocity) == 0:
            continue
        scale_factor = 100 / np.linalg.norm(velocity)
        end_point = (int(center[0] + velocity[0] * scale_factor), int(center[1] - velocity[1] * scale_factor))
        cv2.arrowedLine(frame, center, end_point, (0, 0, 255), 2)



def face_detection(frame, face_cascade):
    global old_positions, velocities, face_labels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    current_centers = {}
    frame_time = 1 / 30  # Assuming a frame rate of 30 FPS; adjust as necessary

    for idx, (x, y, w, h) in enumerate(faces):
        center = (x + w//2, y + h//2)
        current_centers[center] = idx
        color = (0, 0, 255) if idx == 0 else (0, 255, 0)
        face_labels[center] = f"#{idx + 1}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{face_labels[center]} Pos({center[0]}, {center[1]}) Vel({velocities.get(center, (0,0))[0]:.2f}, {velocities.get(center, (0,0))[1]:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Calculate velocity
        if center in old_positions:
            velocity = ((center[0] - old_positions[center][0]) / frame_time, 
                        (center[1] - old_positions[center][1]) / frame_time)
            velocities[center] = velocity
        else:
            velocities[center] = (0, 0)
        old_positions[center] = center

    return frame, current_centers  # Make sure to return both frame and current_centers









def draw_bounding_boxes_and_labels(frame, faces, current_centers):
    for idx, (x, y, w, h) in enumerate(faces):
        center = current_centers[idx]
        color = face_colors.get(center, (255, 0, 0))
        label = face_labels.get(center, f"#{idx+1}")
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)



def update_plot(frame_number, ax1, ax2):
    if positions:
        ax1.plot([pos[0] for pos in positions[-10:]], [pos[1] for pos in positions[-10:]], label='Position')
        ax2.plot([vel[0] for vel in velocities[-10:]], [vel[1] for vel in velocities[-10:]], label='Velocity')
        ax1.legend()
        ax2.legend()
        ax1.set_title('Position over Time')
        ax2.set_title('Velocity over Time')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax2.set_xlabel('X Velocity')
        ax2.set_ylabel('Y Velocity')
        
        

def stream_camera():
    cascade_path = '/home/pi4/Documents/model/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise Exception("Failed to load cascade classifier. Check the path.")

    command = "libcamera-vid -t 0 --inline --listen -o tcp://0.0.0.0:8888"
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Allow libcamera-vid to initialize

    if p.poll() is not None:
        raise Exception("Failed to start libcamera-vid. Error: " + p.stderr.read().decode())

    cap = cv2.VideoCapture("tcp://127.0.0.1:8888")
    if not cap.isOpened():
        raise Exception("Cannot open camera stream.")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    animation = FuncAnimation(fig, update_plot, fargs=(ax1, ax2), interval=1000)
    plt.show(block=False)
    
    # Display the number of detected faces
    frame = display_face_count(frame, faces)

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame, current_centers = face_detection(frame, face_cascade)  # Ensure this is handling returns correctly
            draw_velocity_vectors(frame)  # Corrected function call
            cv2.imshow('Camera Feed with Face Detection', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        p.terminate()
        p.wait()
        plt.close(fig)

        # Save position plot
        plt.figure()
        plt.plot([pos[0] for pos in positions], [pos[1] for pos in positions])
        plt.title('Position over Time')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig('/home/pi4/Documents/positionPlots/position_plot.png')
        plt.close()

        # Save velocity plot
        plt.figure()
        plt.plot([vel[0] for vel in velocities], [vel[1] for vel in velocities])
        plt.title('Velocity over Time')
        plt.xlabel('X Velocity')
        plt.ylabel('Y Velocity')
        plt.savefig('/home/pi4/Documents/velocityPlots/velocity_plot.png')
        plt.close()

if __name__ == "__main__":
    stream_camera()
