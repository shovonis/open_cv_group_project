import os
import random
import cv2
import numpy as np


def frames_extraction(video_path, image_height, image_width):
    # Empty List declared to store video frames
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    while True:
        # Reading a frame from the video file
        success, frame = video_reader.read()
        if not success:
            break
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame[:, 0:256], (image_height, image_width))
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)
    # Closing the VideoCapture object and releasing all resources.
    video_reader.release()
    # returning the frames list
    return frames_list


def create_dataset(classes_list, dataset_directory, max_images_per_class):
    # Declaring Empty Lists to store the features and labels values.
    temp_features = []
    features = []
    labels = []
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        # Getting the list of video files present in the specific class name directory
        files_list = os.listdir(os.path.join(dataset_directory, class_name))
        # Iterating through all the files present in the files list
        for file_name in files_list:
            # Construct the complete video path
            video_file_path = os.path.join(dataset_directory, class_name, file_name)
            # Calling the frame_extraction method for every video file path
            frames = frames_extraction(video_file_path, 128, 128)
            # Appending the frames to a temporary list.
            temp_features.extend(frames)
        # Adding randomly selected frames to the features list
        features.extend(random.sample(temp_features, max_images_per_class))
        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * max_images_per_class)
        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()
    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels
