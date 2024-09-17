import cv2
import os
import concurrent.futures
import numpy as np
import sys

i = 0

# Function to preprocess images
def preprocess_image(image, student_id, face_images):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Apply histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Detect faces in the image
    face_cascade = cv2.CascadeClassifier('E:\smart attendance system project code\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("Number of faces detected: {}".format(len(faces)))
    if len(faces) == 0:
        print("Faces not detected")
        return None
    # Crop and resize the face images
    preprocessed_images = []
    for (x, y, w, h) in faces:
        face_image = gray[y:y+h, x:x+w]
        resized_face_image = cv2.resize(face_image, (64, 64))
        preprocessed_images.append(resized_face_image)
        # Save the preprocessed image to the "dataset" directory
        directory = os.path.join("dataset1", f"student{student_id}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_filename = os.path.join(directory, f"image_{i+len(face_images)}.jpg")
        cv2.imwrite(image_filename, resized_face_image)
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
    except Exception as e:
        print("Error computing HOG:", e)
        return None
    sys.stdout.flush()
    if len(feature_vector) == 0:
        print("Empty feature vector")
    print(feature_vector)
    return feature_vector

# Function to create database
def create_database():
    student_id = 2  # Change this to your student ID
    camera = cv2.VideoCapture(0)
    face_images = []
    while len(face_images) < 100:
        ret, frame = camera.read()
        face_images += preprocess_image(frame, student_id, face_images)
    camera.release()

    # Apply HOG technique using multithreading
    feature_vectors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print("Number of threads created:", executor._max_workers)
        for feature_vector in executor.map(apply_hog, face_images):
            print("Feature vector length after computation:", len(feature_vector))
            feature_vectors.append(feature_vector)
    
    if len(feature_vectors) == 0:
        print("No feature vectors extracted!")
    else:
        # Store the feature vectors
        print("Feature vectors created!")
        directory = 'database/student{}'.format(student_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        feature_vectors_array = np.asarray(feature_vectors)
        print("Number of feature vectors extracted:", len(feature_vectors_array))
        try:
            np.save('{}/features.npy'.format(directory), feature_vectors_array)
            print("Feature vectors saved successfully!")
        except Exception as e:
            print("Error saving feature vectors:", e)

create_database()
