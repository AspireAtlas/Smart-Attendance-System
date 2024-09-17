import cv2
import numpy as np
from scipy.spatial.distance import cosine
import concurrent.futures
import sqlite3
# Function to preprocess images
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Detect faces in the image
    face_cascade = cv2.CascadeClassifier('E:\smart attendance system project code\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("Number of faces detected: {}".format(len(faces)))
    if len(faces) == 0:
        print("No faces detected")
        return None
    # Crop and resize the face images
    preprocessed_images = []
    for (x, y, w, h) in faces:
        face_image = gray[y:y + h, x:x + w]
        resized_face_image = cv2.resize(face_image, (64, 64))
        preprocessed_images.append(resized_face_image)
    print("Number of images read: {}".format(len(preprocessed_images)))
    return preprocessed_images
# Function to apply HOG technique to images
def apply_hog(image):
    winSize = (64, 128)  # Change the window size here
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    if image.shape[0] < winSize[1] or image.shape[1] < winSize[0]:
        image = cv2.resize(image, winSize)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    # Check if the image is in grayscale format
    if len(image.shape) > 2 and image.shape[2] != 1:
        print("Converting image to grayscale")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Check if the image is valid
    if image is None:
        print("Invalid image")
        return None
    try:
        feature_vector = hog.compute(image)
        feature_vector = feature_vector.reshape(-1)
    except Exception as e:
        print("Error computing HOG:", e)
        return None
    if len(feature_vector) == 0:
        print("Empty feature vector")
    print(feature_vector)
    return feature_vector
# Load the database
studentid = 1  # Change this to the student id you want to authenticate
directory = 'database/student{}'.format(studentid)
features = np.load('{}/features.npy'.format(directory))
# Create the attendance dictionary
attendance = {}
# Capture video from the camera
camera = cv2.VideoCapture(0)
# Function to process a frame and perform authentication
def process_frame(frame):
    # Preprocess the frame
    preprocessed_images = preprocess_image(frame)
    # Extract features from the preprocessed images
    feature_vectors = []
    for preprocessed_image in preprocessed_images:
        feature_vector = apply_hog(preprocessed_image)
        if feature_vector is not None:
            feature_vectors.append(feature_vector)
    # Calculate the distance between the extracted features and the features in the database
    distances = []
    for feature_vector in feature_vectors:
        if feature_vector is not None:
            feature_vector = feature_vector.flatten()
            resized_features = np.resize(features, feature_vector.shape)
            distance = cosine(feature_vector, resized_features)
            distances.append(distance)
    threshold = 0.15  # Change this to your desired threshold
    if len(distances) > 0:
        mean_distance = np.mean(distances)
        print("Mean Distance:", mean_distance)

        # Authenticate user based on the mean distance
        if mean_distance <= threshold:
            # User authenticated
            print("student matched!")
            # Add the user to the attendance dictionary
            attendance[studentid] = True
        else:
            # User not authenticated
            print("student not matched!")
            # Add the user to the attendance dictionary
            attendance[studentid] = False
    else:
        print("No faces detected.")
# Function to store attendance in the database
def store_attendance():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                      (userid INTEGER PRIMARY KEY, authenticated INTEGER)''')
    print("database is created!!")
    # Insert attendance records
    for studentid, authenticated in attendance.items():
        cursor.execute("REPLACE INTO attendance VALUES (?, ?)", (studentid, int(authenticated)))

    # Commit changes and close connection
    conn.commit()
    conn.close()
# Process frames using multithreading
with concurrent.futures.ThreadPoolExecutor() as executor:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        # Submit frame processing task to the executor
        executor.submit(process_frame, frame)
        # Display the frame
        cv2.imshow("Attendance System", frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release the video capture
camera.release()
# Print the attendance dictionary
print("Attendance:")
for userid, authenticated in attendance.items():
    print("student {}: {}".format(studentid, "Present" if authenticated else "Absent"))
# Store attendance in the database
store_attendance()
