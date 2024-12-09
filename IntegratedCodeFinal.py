import tkinter as tk
import serial
import json
import time
import math
from tkinter import simpledialog, messagebox, OptionMenu, Scale, ttk
from PIL import Image, ImageTk
import os
import webbrowser
import threading
from flask import Flask, render_template
from werkzeug.serving import run_simple
import torch
import cv2
import numpy as np
import signal
import sys



#Paths
model_path = "/home/praneethboddu/Downloads/capstone/faceModels/yolov5nano.pt"
yolov5_repo_path = "/home/praneethboddu/Downloads/capstone/yolov5"

# Load the YOLOv5 model
model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path, source='local')
model.eval()  # Set the model to evaluation mode


# Confidence threshold for detections
conf_threshold = 0.5





# Flask app definition
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/gimbal')
def gimbal():
    return render_template('gimbal.html')

@app.route('/rail')
def rail():
    return render_template('rail.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

def run_flask():
    run_simple('127.0.0.1', 5000, app)
'''
def save_positions(rail_position, gimbal_position):
    positions = {
        "rail_position": rail_position,
        "gimbal_position": gimbal_position
    }
    with open("positions.json", "w") as file:
        json.dump(positions, file)

def load_positions():
    try:
        with open("positions.json", "r") as file:
            positions = json.load(file)
            return positions["rail_position"], positions["gimbal_position"]
    except FileNotFoundError:
            # Return default positions if file does not exist
        return 0, 0  # Assuming 0 is the default position

# Function to send position to Arduino
def send_position_to_arduino(position):
    # Open serial connection to Arduino
    arduino = serial.Serial('/dev/ttyACM0', 9600)  # Adjust port if necessary
    arduino.write(str(position).encode())  # Send position as byte string
    arduino.close()
'''   
class UserInterface:
    exit1 = False
    
    def __init__(self, master):
        self.master = master
        master.title("CarPlay Interface")
        master.geometry("800x480")  # Set the window size to match a typical CarPlay display

        # Load background image
        bg_image = Image.open("landscape.jpg")
        bg_image = bg_image.resize((800, 480), Image.Resampling.LANCZOS)
        self.background_image = ImageTk.PhotoImage(bg_image)

        # Create canvas to display background image
        self.canvas = tk.Canvas(master, width=800, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Display background image on canvas
        self.canvas.create_image(0, 0, image=self.background_image, anchor=tk.NW)

         # Load the images for the camera button
        self.camera_on_icon = Image.open("camera_on_icon.png").resize((150, 150), Image.Resampling.LANCZOS)
        self.camera_off_icon = Image.open("camera_off_icon.png").resize((150, 150), Image.Resampling.LANCZOS)

        # Convert images to PhotoImage objects
        self.camera_on_image = ImageTk.PhotoImage(self.camera_on_icon)
        self.camera_off_image = ImageTk.PhotoImage(self.camera_off_icon)

        # Initial state of the camera
        self.camera_on = True

        # Camera button with the initial image
        self.camera_button = tk.Button(self.canvas, image=self.camera_on_image, command=self.toggle_camera_buttons, bd=0)
        self.camera_button.place(x=300, y=170)
        self.camera_button.config(text="Camera")
        '''
         # Microphone State and Icons
        self.microphone_on = True
        microphone_on_icon = Image.open("microphone_on_icon.png")
        microphone_off_icon = Image.open("microphone_off_icon.png")
        self.microphone_on_icon = ImageTk.PhotoImage(microphone_on_icon.resize((150, 150), Image.ANTIALIAS))
        self.microphone_off_icon = ImageTk.PhotoImage(microphone_off_icon.resize((150, 150), Image.ANTIALIAS))

        # Microphone Button
        self.microphone_button = tk.Button(self.canvas, image=self.microphone_on_icon, command=self.toggle_microphone, bd=0)
        self.microphone_button.place(x=300, y=170)
        self.microphone_button.config(text="Microphone")
        '''
        # Gimbal Icon
        gimbal_icon = Image.open("gimbal_icon.png")
        gimbal_icon = gimbal_icon.resize((150, 150), Image.Resampling.LANCZOS)
        self.gimbal_image = ImageTk.PhotoImage(gimbal_icon)
        self.gimbal_button = tk.Button(self.canvas, image=self.gimbal_image, command=self.toggle_gimbal_system, bd=0)
        self.gimbal_button.place(x=100, y=170)
        self.gimbal_button.config(text="Gimbal")

        # Rail System Icon
        rail_icon = Image.open("rail_icon.png")
        rail_icon = rail_icon.resize((150, 150), Image.Resampling.LANCZOS)
        self.rail_image = ImageTk.PhotoImage(rail_icon)
        self.rail_button = tk.Button(self.canvas, image=self.rail_image, command=self.toggle_rail_system, bd=0)
        self.rail_button.place(x=500, y=170)
        self.rail_button.config(text="Rail System")
        '''
        # Trumpet Icon
        trumpet_icon = Image.open("trumpet_icon.jpg")
        trumpet_icon = trumpet_icon.resize((35, 35), Image.ANTIALIAS)
        self.trumpet_image = ImageTk.PhotoImage(trumpet_icon)
        self.trumpet_button = tk.Button(self.canvas, image=self.trumpet_image, command=self.trumpet_controls, bd=0)
        self.trumpet_button.place(x=700, y=400)
        self.trumpet_button.config(text="Trumpet")
        self.trumpet_button.image = trumpet_icon  # Keep a reference to avoid garbage collection
        #self.trumpet_button.pack()

        # Load saved volume or set to default (50)
        self.volume_file = "volume_memory.txt"
        self.current_volume = self.load_volume()


        # Create open camera button (hidden initially)
        #self.open_camera_button = tk.Button(self.canvas, text="Open Camera", font=("Helvetica", 12), command=self.open_camera)
        #self.open_camera_button.place(x=400, y=280)
        #self.open_camera_button.place_forget()  # Hide open camera button initially

        # Create close camera button (hidden initially)
        #self.close_camera_button = tk.Button(self.canvas, text="Close Camera", font=("Helvetica", 12), command=self.close_camera)
        #self.close_camera_button.place(x=400, y=320)
        #self.close_camera_button.place_forget()  # Hide close camera button initially

        # Initialize microphone buttons
        self.mute_button = None
        '''

        # Initialize gamepad (hidden initially)
        self.gimbal_visible = False
        self.gimbal_system = GimbalSystem(master)
        

        # Initialize rail buttons
        self.turn_left_button = None
        self.turn_right_button = None

        # Initialize microphone volume slider (hidden initially)
        self.microphone_controls_visible = False

        # Initialize information components
        self.create_information_components()

        # Dictionary to hold pop=up windows for information components
        self.popup_windows = {}

        # Initialize trumpet controls (hidden initially)
        self.trumpet_controls_visible = False
        self.trumpet_slider = None

        # Initialize rail system (hidden initially)
        self.rail_visible = False
        self.rail_system = RailSystem(master)

        # Start/Stop Tracking Button
        self.start_button = tk.Button(master, text="Start Tracking", command=self.start_tracking, width=15, height=10)
        self.start_button.place(x=650, y=170)

         # Load saved positions
        #self.rail_position, self.gimbal_position = load_positions()
        
        # Close event to save positions before closing
        #master.protocol("WM_DELETE_WINDOW", self.close_program)

    '''
    def close_program(self):
        # Save the current positions when closing
        save_positions(self.rail_position, self.gimbal_position)
        self.master.quit()  # Close the program
    '''


    def faceTracking(self):
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)
        #signal.signal(signal.SIGINT, self.cleanup_and_exit(self, sig, frame))  # Handle Ctrl+C

        # Open the camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            #sys.exit(1)
        else:
            print("Camera Opened!")
            self.start_button.config(text="Stop Tracking", command=self.stop_tracking) 
        
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mid_x, mid_y = frame_width // 2, frame_height // 2  # Center of the frame
        #print("Exit: ", self.exit1)
        
        while (not self.exit1):
           
            #print("Stop Tracking Button Displayed")
            #self.start_button.config(text="Stop Tracking", command=self.stop_tracking)
            
            
            #cleanup_and_exit(sig, frame)
            
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            else:
                # Show the video feed
                frame = cv2.flip(frame, 0)
                #cv2.imshow("Live Video Feed", frame)
            
                # Performing Face Tracking
                results = model(frame)
                detections = results.pred[0]
                faces = []
                
                for det in detections:
                    confidence = det[4].item()
                    if confidence > conf_threshold:
                        x_min, y_min, x_max, y_max = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                        faces.append((x_min, y_min, x_max - x_min, y_max - y_min))
                      
                # Calculate the average x coordinate of the detected faces
                if faces:
                    avg_x = np.mean([x + w / 2 for x, y, w, h in faces])
                    avg_y = np.mean([y + h / 2 for x, y, w, h in faces])
                    print(f"Detected face at (x, y): ({avg_x}, {avg_y})")
                    
                    # Adjust camera based on average x position relative to the frame center
                    if avg_x > mid_x * 1.2:  # Move camera to the right if >20% deviation
                        print("Moving camera right")
                        tempCommand = 'B'
                        arduino.write(tempCommand.encode())
                        while (avg_x > mid_x * 1.1 and not self.exit1):  # Continue until <10% deviation
                            ret, frame = cap.read()
                            frame = cv2.flip(frame, 0)
                            #cv2.imshow("Live Video Feed", frame)
                            tempCommand = 'B'
                            arduino.write(tempCommand.encode())
                            if not ret:
                                break
                            results = model(frame)
                            detections = results.pred[0]
                            faces = [(int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]))
                                     for det in detections if det[4].item() > conf_threshold]
                            if faces:
                                avg_x = np.mean([x + w / 2 for x, y, w, h in faces])
                                avg_y = np.mean([y + h / 2 for x, y, w, h in faces])
                                print(f"Updated face position: ({avg_x}, {avg_y}) - Moving right")
                            else:
                                break  # Break if no faces are detected
                    elif avg_x < mid_x * 0.8:  # Move camera to the left if >20% deviation
                        print("Moving camera left")
                        tempCommand = 'F'
                        arduino.write(tempCommand.encode())
                        while (avg_x < mid_x * 0.9 and not self.exit1):  # Continue until <10% deviation
                            ret, frame = cap.read()
                            frame = cv2.flip(frame, 0)
                            #cv2.imshow("Live Video Feed", frame)
                            print("Moving camera left")
                            tempCommand = 'F'
                            arduino.write(tempCommand.encode())
                            if not ret:
                                break
                            results = model(frame)
                            detections = results.pred[0]
                            faces = [(int(det[0]), int(det[1]), int(det[2]) - int(det[0]), int(det[3]) - int(det[1]))
                                     for det in detections if det[4].item() > conf_threshold]
                            if faces:
                                avg_x = np.mean([x + w / 2 for x, y, w, h in faces])
                                avg_y = np.mean([y + h / 2 for x, y, w, h in faces])
                                print(f"Updated face position: ({avg_x}, {avg_y}) - Moving left")
                            else:
                                break  # Break if no faces are detected
                    else:
                        tempCommand = 'S'
                        arduino.write(tempCommand.encode())
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
            
            
        print("Exiting Face Tracking!")
        # Release resources
        sys.exit(0)
        cap.release()
        cv2.destroyAllWindows()
        
        
    '''
    # Exit handler
    def cleanup_and_exit(self, sig, frame):
        print("Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)
    '''


    def toggle_rail_system(self):
        if not self.rail_visible:
            # Show rail system
            self.rail_system.show()
            self.rail_visible = True
        else:
            # Hide rail system
            self.rail_system.hide()
            self.rail_visible = False

    def toggle_camera_buttons(self):
        if self.camera_on:
            self.camera_image = ImageTk.PhotoImage(self.camera_off_icon)
            print("Camera Off")
        else:
            self.camera_image = ImageTk.PhotoImage(self.camera_on_icon)
            print("Camera On")

        # Update button image and toggle state
        self.camera_button.config(image=self.camera_image)
        self.camera_on = not self.camera_on

    def start_tracking(self):
        self.exit1 = False
        # Hide the icons for rail, gimbal, and camera
        self.rail_button.place_forget()
        self.gimbal_button.place_forget()
        self.camera_button.place_forget()
        # Change the "Start Tracking" button to "Stop Tracking"
        self.start_button.config(text="Stop Tracking", command=self.stop_tracking)
        print("Start Tracking Pressed!")
        #self.faceTracking()
        threading.Thread(target=self.faceTracking, daemon=True).start()
        #self.rail_scale.pack_forget()
        #self.gimbal_scale.pack_forget()

        

    def stop_tracking(self):
        self.exit1 = True
        # Show the icons for rail, gimbal, and camera
        self.gimbal_button.place(x=100, y=170)
        self.rail_button.place(x=500, y=170)
        self.camera_button.place(x=300, y=170)
        print("Stop Tracking Pressed!")
        # Change the "Stop Tracking" button to "Start Tracking"
        self.start_button.config(text="Start Tracking", command=self.start_tracking)
        #self.rail_scale.pack(pady=10)
        #self.gimbal_scale.pack(pady=10)
        
        #self.cleanup_and_exit(self, sig, frame)
        
        
    
    #def open_camera(self):
     #   print("Opening camera")

    #def close_camera(self):
     #   print("Closing camera")
    '''
    def toggle_microphone(self):
        self.microphone_on = not self.microphone_on
        if self.microphone_on:
            self.microphone_button.config(image=self.microphone_on_icon)
            print("Microphone On")
        else:
            self.microphone_button.config(image=self.microphone_off_icon)
            print("Microphone Off")

    
    #def toggle_mute(self):
        # Toggle mute state (implement desired behavior)
     #   print("Mute")
      #  pass

    def load_volume(self):
        # Load the last saved volume from file or return default value (50)
        if os.path.exists(self.volume_file):
            with open(self.volume_file, "r") as f:
                return int(f.read().strip())
        return 50

    def save_volume(self):
        # Save the current volume to file
        with open(self.volume_file, "w") as f:
            f.write(str(self.current_volume))

    def trumpet_controls(self):
        if self.trumpet_controls_visible:
            # Hide trumpet controls
            if self.trumpet_slider:
                self.trumpet_slider.destroy()
                self.trumpet_slider = None
            self.trumpet_controls_visible = False
        else:
            # Show trumpet controls with a gradient from 0 (bottom) to 100 (top)
            self.trumpet_slider = ttk.Scale(self.master, from_=100, to=0, orient=tk.VERTICAL, length=200,
                                            command=self.update_volume)
            self.trumpet_slider.set(self.current_volume)
            self.trumpet_slider.place(x=700, y=150)
            self.trumpet_controls_visible = True

    def update_volume(self, value):
        # Update the current volume and print it
        self.current_volume = int(float(value))
        print(f"Current volume: {self.current_volume}")
        self.save_volume()  # Save the updated volume to file
    '''
    def toggle_gimbal_system(self):
        if not self.gimbal_visible:
              # Show gimbal system
            self.gimbal_system.show()
            self.gimbal_visible = True
        else:
              # Hide gimbal system
            self.gimbal_system.hide()
            self.gimbal_visible = False
                
    #def toggle_gamepad(self):
        #if not self.gamepad_visible:
            # Show gamepad
          #  self.gamepad.show()
          #  self.gamepad_visible = True
       # else:
            # Hide gamepad
         #   self.gamepad.hide()
         #   self.gamepad_visible = False


   # def hide_camera_buttons(self):
    #    if self.open_camera_button:
     #       self.open_camera_button.place_forget()
      #  if self.close_camera_button:
       #     self.close_camera_button.place_forget()

    #def hide_microphone_buttons(self):
     #   if self.mute_button:
      #      self.mute_button.place_forget()
       

    def hide_rail_buttons(self):
        if self.turn_left_button:
            self.turn_left_button.place_forget()
        if self.turn_right_button:
            self.turn_right_button.place_forget()

    def create_information_components(self):
        # Information options for selection
        info_options = ["About Us"]


        

        # Create about us selection box
        button_about_us = tk.Button(self.canvas, text="About Us", font=("Helvetica", 14), bg="white",
                                      command=lambda: self.show_info_popup("About Us"))
        button_about_us.place(x=1, y=1)
        

    def show_info_popup(self, info_type):
    # Check if pop-up window for this info type already exists
        if info_type in self.popup_windows:
            popup_window = self.popup_windows[info_type]
            if popup_window.winfo_exists():
                popup_window.focus()  # Bring existing window to front
                return
            else:
                del self.popup_windows[info_type]  # Remove closed window from dictionary


        # Create a new pop-up window for the selected info type
        popup_window = tk.Toplevel(self.master)
        popup_window.title(info_type)
        popup_window.geometry("300x200+550+100")  # Size and position of the pop-up window
        '''
        # Display different content based on the button clicked
        if info_type == "Notifications":
            message_label = tk.Label(popup_window, text="No new notifications", font=("Helvetica", 14))
            message_label.pack(pady=20)

        elif info_type == "Warnings":
            message_label = tk.Label(popup_window, text="You have no new warnings", font=("Helvetica", 14))
            message_label.pack(pady=20)

            # Example of a mailbox-like structure
            mailbox_frame = tk.Frame(popup_window)
            mailbox_frame.pack(pady=10)
            
            for i in range(3):  # Example of three warning messages
                warning_message = tk.Label(mailbox_frame, text=f"Warning {i+1}: This is a warning message.", font=("Helvetica", 12))
                warning_message.pack()

        elif info_type == "Tooltips":
            webbrowser.open_new("http://127.0.0.1:5000/")
            #tooltip_label = tk.Label(popup_window, text="Helpful Links:", font=("Helvetica", 14))
            #tooltip_label.pack(pady=10)


            # Example links
            #gimbal_link = tk.Label(popup_window, text="How to use the Gimbal", font=("Helvetica", 12), fg="blue", cursor="hand2")
            #gimbal_link.pack()
            #gimbal_link.bind("<Button-1>", lambda e: self.open_link("http://127.0.0.1:5000/"))

            #rail_link = tk.Label(popup_window, text="How to use the Rail", font=("Helvetica", 12), fg="blue", cursor="hand2")
            #rail_link.pack()
            #rail_link.bind("<Button-1>", lambda e: self.open_link("https://example.com/rail"))

            #camera_link = tk.Label(popup_window, text="How to use the Camera", font=("Helvetica", 12), fg="blue", cursor="hand2")
            #camera_link.pack()
            #camera_link.bind("<Button-1>", lambda e: self.open_link("https://example.com/camera"))

        '''
        if info_type == "About Us":
            webbrowser.open_new("http://127.0.0.1:5000/")
            #about_label = tk.Label(popup_window, text="About Us:\nThis is an example application.", font=("Helvetica", 14))
            #about_label.pack(pady=20)

        # Store reference to the pop-up window
        self.popup_windows[info_type] = popup_window

    def open_link(self, url):
        import webbrowser
        webbrowser.open(url)

    def close_info_popup(self, info_type):
        # Close the pop-up window for the specified info type
        if info_type in self.popup_windows:
            self.popup_windows[info_type].destroy()  # Destroy the window
            del self.popup_windows[info_type]  # Remove reference from dictionary
    

class Gamepad:
    def __init__(self, master, radius=50):
        self.master = master
        self.radius = radius
        self.canvas = tk.Canvas(master, width=2*self.radius, height=2*self.radius, bg='white')

        # Create circular gamepad (initially hidden)
        pad_size = 2 * self.radius
        self.circle = self.canvas.create_oval((pad_size - 2*self.radius) / 2, (pad_size - 2*self.radius) / 2,
                                               (pad_size + 2*self.radius) / 2, (pad_size + 2*self.radius) / 2,
                                               outline='black', state=tk.HIDDEN)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.handle_drag)
        self.canvas.bind("<ButtonRelease-1>", self.handle_release)

        # Initialize mouse position
        self.prev_x = 0
        self.prev_y = 0

    def show(self):
        # Show canvas and gamepad circle
        self.canvas.place(x=30, y=340, anchor=tk.W)  # Position canvas on the left side of the screen
        self.canvas.itemconfigure(self.circle, state=tk.NORMAL)

    def hide(self):
        # Hide canvas and gamepad circle
        self.canvas.place_forget()
        self.canvas.itemconfigure(self.circle, state=tk.HIDDEN)

    def handle_drag(self, event):
        # Calculate current mouse position relative to the center of the circle
        x = event.x
        y = event.y

        # Calculate distance from center of the circle
        distance = math.sqrt((x - self.radius)**2 + (y - self.radius)**2)

        if distance <= self.radius:
            # Calculate angle of the mouse position relative to the positive x-axis
            angle = math.atan2(self.radius - y, x - self.radius)

            # Convert angle to degrees
            degrees = math.degrees(angle)

            # Determine direction based on angle
            if -45 <= degrees < 45:
                direction = "right"
            elif 45 <= degrees < 135:
                direction = "up"
            elif -135 <= degrees < -45:
                direction = "down"
            else:
                direction = "left"

            # Apply weight based on distance
            # Angle varies from 0 (center) to 90 (edge of circle)
            tilt_angle = (distance / self.radius) * 90

            # Print direction and tilt angle
            print(f"Tilting {direction} by {tilt_angle:.2f} degrees")

            # Update previous mouse position
            self.prev_x = x
            self.prev_y = y

            
class GimbalSystem:
    
    def handle_release(self, event):
        # Reset previous mouse position on release
        self.prev_x = 0
        self.prev_y = 0

    def __init__(self, master):
        self.master = master
        self.buttons_visible = False

       # Load arrow images (replace these with your own arrow images)
        arrow_up_icon = Image.open("arrow_up.jpg")
        arrow_up_icon = arrow_up_icon.resize((50, 50))
        self.arrow_up_image = ImageTk.PhotoImage(arrow_up_icon)
        
        
        arrow_down_icon = Image.open("arrow_down.jpg")
        arrow_down_icon = arrow_down_icon.resize((50, 50))
        self.arrow_down_image = ImageTk.PhotoImage(arrow_down_icon)

       
        

        # Create arrow buttons (initially hidden)
        self.button_up = tk.Button(self.master, image=self.arrow_up_image, bg="white", bd=0)
        self.button_up.bind("<ButtonPress>", lambda event: self.start_moving('up'))
        self.button_up.bind("<ButtonRelease>", lambda event: self.stop_moving())

        self.button_down = tk.Button(self.master, image=self.arrow_down_image, bg="white", bd=0)
        self.button_down.bind("<ButtonPress>", lambda event: self.start_moving('down'))
        self.button_down.bind("<ButtonRelease>", lambda event: self.stop_moving())



    def show(self):
        # Show arrow buttons
        self.button_up.place(x=50, y=180)
        self.button_down.place(x=50, y=280)

        self.buttons_visible = True

    def hide(self):
        # Hide arrow buttons
        self.button_up.place_forget()
        self.button_down.place_forget()
        self.buttons_visible = False

    def start_moving(self, direction):
        self.moving = True
        self.move_direction = direction
        self.move_rail()

    def stop_moving(self):
        self.moving = False

    def move_rail(self):
        if self.moving:
            if self.move_direction == 'up':
                self.move_up()
            elif self.move_direction == 'down':
                self.move_down()
            

            # Continue moving after 100ms if the button is still pressed
            self.master.after(100, self.move_rail)

    def move_up(self):
        print("Tilting camera up")

    def move_down(self):
        print("Tilting camera down")

    

        
class RailSystem:
    def __init__(self, master):
        self.master = master
        self.buttons_visible = False
        # Adjust the COM part
        self.arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)

       # Load arrow images (replace these with your own arrow images)
        arrow_up_icon = Image.open("arrow_up.jpg")
        arrow_up_icon = arrow_up_icon.resize((50, 50))
        self.arrow_up_image = ImageTk.PhotoImage(arrow_up_icon)
        
        
        arrow_down_icon = Image.open("arrow_down.jpg")
        arrow_down_icon = arrow_down_icon.resize((50, 50))
        self.arrow_down_image = ImageTk.PhotoImage(arrow_down_icon)

        
        arrow_left_icon = Image.open("arrow_left.jpg")
        arrow_left_icon = arrow_left_icon.resize((50, 50))
        self.arrow_left_image = ImageTk.PhotoImage(arrow_left_icon)
        
        
        arrow_right_icon = Image.open("arrow_right.jpg")
        arrow_right_icon = arrow_right_icon.resize((50, 50))
        self.arrow_right_image = ImageTk.PhotoImage(arrow_right_icon)
       
        

        # Create arrow buttons (initially hidden)
        #self.button_up = tk.Button(self.master, image=self.arrow_up_image, bg="white", bd=0)
        #self.button_up.bind("<ButtonPress>", lambda event: self.start_moving('up'))
        #self.button_up.bind("<ButtonRelease>", lambda event: self.stop_moving())

        #self.button_down = tk.Button(self.master, image=self.arrow_down_image, bg="white", bd=0)
        #self.button_down.bind("<ButtonPress>", lambda event: self.start_moving('down'))
        #self.button_down.bind("<ButtonRelease>", lambda event: self.stop_moving())

        self.button_left = tk.Button(self.master, image=self.arrow_left_image, bg="white", bd=0)
        self.button_left.bind("<ButtonPress>", self.on_left_press)
        self.button_left.bind("<ButtonRelease>", self.on_button_release)

        self.button_right = tk.Button(self.master, image=self.arrow_right_image, bg="white", bd=0)
        self.button_right.bind("<ButtonPress>", self.on_right_press)
        self.button_right.bind("<ButtonRelease>", self.on_button_release)

    def show(self):
        # Show arrow buttons
        #self.button_up.place(x=580, y=180)
        #self.button_down.place(x=580, y=280)
        self.button_left.place(x=520, y=100)
        self.button_right.place(x=640, y=100)
        self.buttons_visible = True

    def hide(self):
        # Hide arrow buttons
        #self.button_up.place_forget()
        #self.button_down.place_forget()
        self.button_left.place_forget()
        self.button_right.place_forget()
        self.buttons_visible = False

    def send_command(self, command):
        """Send a command to the Arduino."""
        self.arduino.write(command.encode())
        time.sleep(0.1)  # Delay for communication

    def on_left_press(self, event):
        """Handle the left button press (Forward)."""
        self.send_command('F')

    def on_right_press(self, event):
        """Handle the right button press (Backward)."""
        self.send_command('B')

    def on_button_release(self, event):
        """Stop the motor when the button is released."""
        self.send_command('S')


'''
    def start_moving(self, direction):
        self.moving = True
        self.move_direction = direction
        self.move_rail()

    def stop_moving(self):
        self.moving = False
        self.send_command('STOP')

    def move_rail(self):
        if self.moving:
            #if self.move_direction == 'up':
             #   self.move_up()
            #elif self.move_direction == 'down':
             #   self.move_down()
            if self.move_direction == 'LEFT':
                self.move_left()
    
            elif self.move_direction == 'RIGHT':
                self.move_right()

            # Continue moving after 100ms if the button is still pressed
            self.master.after(100, self.move_rail)

    #def move_up(self):
     #   self.send_command('UP')
      #  print("Moving rail up")

    #def move_down(self):
     #   self.send_command('DOWN')
      #  print("Moving rail down")

    def move_left(self):
        #self.send_command('LEFT')
        print("Moving rail left")

    def move_right(self):
        #self.send_command('RIGHT')
        print("Moving rail right")

    def send_command(self, command):
        self.arduino.write((command + '\n').encode())  # Send the command to Arduino
        time.sleep(0.1)  # Small delay for serial communication
'''

def main():
    root = tk.Tk()
    ui = UserInterface(root)
    #ui.show()
    root.mainloop()
    #ui.arduino.close()


if __name__ == "__main__":
    main()
    #app.run(debug=True)
