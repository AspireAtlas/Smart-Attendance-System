
# Smart Attendance System Using Face Recognition

Our project is a Face Attendance System that uses computer vision techniques to detect and recognize faces in real-time. It captures images from a camera, preprocesses them, and applies the Histogram of Oriented Gradients (HOG) technique to extract features. The extracted features are then compared to the features stored in a database to authenticate the faces.


## Features
• Real-time face detection and recognition

• Attendance tracking and recording

• Flexible threshold for authentication

• Multithreading for efficient processing

## Requirements
•	Python 3.6+ 

•	OpenCV

•	NumPy

•	SQLite (for attendance storage)


## Setup
####  1.	Install the required dependencies:

    pip install opencv-python numpy
#### 2. Download the Haar Cascade classifier file:
  •	Visit the OpenCV GitHub repository and navigate to the haarcascades directory.

  • Download the haarcascade_frontolface_default.xml file and place it in the project directory.

#### 3. Capture and preprocess student images:
   • Run the create_database function in the database.py file.

   • Adjust the student_id variable and the number of captured images as needed. 

   • This will create a directory for the student and store the preprocessed images and feature vectors in the database directory.

 ####  4. Run the Face Attendance System:
    python face_attendance_system.py

#### 5. Press 'q' to exit the program.

## Configuration
Adjust the threshold value in the process_frame function of face_attendance_system.py to modify the authentication sensitivity.
## Attendance Storage
•	The attendance records are stored in an SQLite database named attendance.db.

•	The attendance table in the database stores the user ID and the authentication status (0 for absent, 1 for present).

•	The store_attendance function in face_attendance_system.py is responsible for storing the attendance records in the database.
