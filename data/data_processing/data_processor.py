import datetime
from os import walk
import pandas as pd
import cv2
import shutil, os
from src.data_processing.disparity import disparity
from src.data_processing.optical_flow import optical_flow
import numpy as np
import uuid
import matplotlib.pyplot as plt


def read_file(file_name, time_format, date_time_required=True):
    data = pd.read_csv(file_name)
    if date_time_required:
        data['Time'] = pd.to_datetime(data['Time'], format=time_format)
    else:
        data['Time'] = pd.to_datetime(data['Time'], format=time_format).dt.strftime('%H-%M-%S')

    return data


def get_image_file_names(image_dir):
    _, _, filenames = next(walk(image_dir))
    return filenames


def search_a_frame(frames, frame_number):
    matching = [match for match in frames if "Frame-" + str(frame_number) in match]
    return matching[0]


def get_frame_and_time_of_interest(frames, frame_number, matched_frames, window_size=15):
    # Pre Processing
    time_of_frame = matched_frames.replace("Frame-" + str(frame_number) + "-", '')
    time_of_frame = time_of_frame.replace(".png", '')
    time_of_frame = datetime.datetime.strptime(time_of_frame, '%H-%M-%S-%f').strftime('%H-%M-%S')
    time_of_frame = datetime.datetime.strptime(time_of_frame, "%H-%M-%S")

    # Get time + window and time - window
    frames_at_time_plus_window = (time_of_frame + datetime.timedelta(0, 3))
    frames_at_time_minus_window = (time_of_frame + datetime.timedelta(0, -window_size))
    time_of_interests = []

    end_time = time_of_frame
    while frames_at_time_minus_window < end_time:
        time_of_interests.append(frames_at_time_minus_window.time().strftime('%H-%M-%S'))
        frames_at_time_minus_window += datetime.timedelta(0, 1)

    start_time = time_of_frame
    time_of_interests.append(start_time.time().strftime('%H-%M-%S'))
    while start_time < frames_at_time_plus_window:
        start_time += datetime.timedelta(0, 1)
        # print("Start Time + t", current_frame_time.time().strftime('%H-%M-%S'))
        time_of_interests.append(start_time.time().strftime('%H-%M-%S'))

    frame_of_interests = []
    for time in time_of_interests:
        for frame in frames:
            if time in frame:
                frame_of_interests.append(frame)

    return frame_of_interests, time_of_interests


def create_and_save_video(video_name, frames, image_directory):
    video = cv2.VideoWriter(video_name, 0, 22, frame_size)
    for frame in frames:
        current_im_dir = image_directory + '/' + frame
        video.write(cv2.imread(current_im_dir))

    cv2.destroyAllWindows()
    video.release()


def create_disparity_video(video_path, stereo_images, image_directory):
    video = cv2.VideoWriter(video_path, 0, fps, (256, 256))
    for frame in stereo_images:
        current_im_dir = image_directory + '/' + frame
        # print("Image Dir: " + current_im_dir)

        img = cv2.imread(current_im_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        disp_img = disparity.generate_disparity_sgm(img)

        # plt.imshow(disp_img, 'gray')
        # plt.show()

        disp_img = np.uint8(255 * disp_img)
        video.write(disp_img)

    cv2.destroyAllWindows()
    video.release()


def save_frames(dest, frames, image_directory):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for frame in frames:
        current_im_dir = image_directory + '/' + frame
        shutil.copy(current_im_dir, dest)


def process_data_per_class(data_src_path, data_save_path):
    # Get all Frames
    image_dir = data_src_path + 'Frames'
    frame_list = get_image_file_names(image_dir)

    # Get Verbal Feedback
    verbal_feedback_file = data_src_path + '/' + 'verbal_feedback.csv'
    verbal_feedbacks = read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')

    eye_tracking_data = read_file(data_src_path + '/' + 'eye_tracking.csv', time_format='%H-%M-%S-%f',
                                  date_time_required=False)
    head_tracking_data = read_file(data_src_path + '/' + 'head_tracking.csv', time_format='%H-%M-%S-%f',
                                   date_time_required=False)

    # Make Sure to create the data save directory
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    os.makedirs(data_save_path + '/clips', exist_ok=True)
    os.makedirs(data_save_path + '/disp', exist_ok=True)
    os.makedirs(data_save_path + '/optic', exist_ok=True)
    os.makedirs(data_save_path + '/eye', exist_ok=True)
    os.makedirs(data_save_path + '/head', exist_ok=True)

    for index, verbal_feedback_frames in verbal_feedbacks.iterrows():
        frame_at_time_t = verbal_feedback_frames['Frame']
        verbal_feedback = verbal_feedback_frames['CS']
        # Get the frame and Time of Interest
        matched_frame = search_a_frame(frame_list, frame_at_time_t)
        frame_of_interest, time_of_interest = get_frame_and_time_of_interest(frame_list, frame_at_time_t, matched_frame,
                                                                             window_size=window)
        class_directory = ''
        if int(verbal_feedback) <= class_rule["low"]:
            class_directory = '/low'

        if class_rule['low'] < int(verbal_feedback) <= class_rule['medium']:
            class_directory = '/medium'
        if class_rule['medium'] < int(verbal_feedback) <= class_rule['high']:
            class_directory = '/high'

        # Generate Unique File Identifier
        unique_id = str(uuid.uuid4())[:16]

        # Save the video clips
        clips_dir = data_save_path + '/clips/' + class_directory
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)
        video_name = clips_dir + '/clip-' + unique_id + '.mp4'
        create_and_save_video(video_name, frame_of_interest, image_directory=image_dir)

        # Save Optical  Flow
        optic_dir = data_save_path + '/optic/' + class_directory
        if not os.path.exists(optic_dir):
            os.makedirs(optic_dir)
        save_dir = optic_dir + '/disp-' + unique_id + '.mp4'
        optical_flow.get_optical_flow(video_name, save_dir)

        # Save Disparity Map
        disp_dir = data_save_path + '/disp/' + class_directory
        if not os.path.exists(disp_dir):
            os.makedirs(disp_dir)
        video_name = disp_dir + '/disp-' + unique_id + '.mp4'
        create_disparity_video(video_name, frame_of_interest, image_directory=image_dir)

        # Save Eye_tracking
        eye_dir = data_save_path + '/eye/' + class_directory
        if not os.path.exists(eye_dir):
            os.makedirs(eye_dir)
        file = eye_dir + '/eye-' + unique_id + '.csv'
        data = eye_tracking_data[eye_tracking_data['Time'].isin(time_of_interest)]
        data.to_csv(file, index=False)

        # # # Save Head Tracking
        head_dir = data_save_path + '/head/' + class_directory
        if not os.path.exists(head_dir):
            os.makedirs(head_dir)
        file = head_dir + '/head-' + unique_id + '.csv'
        data = head_tracking_data[head_tracking_data['Time'].isin(time_of_interest)]
        data.to_csv(file, index=False)

    return


def process_data_per_individual(data_src_path, data_save_path):
    # Get all Frames
    image_dir = data_src_path + '/Frames'
    frame_list = get_image_file_names(image_dir)

    # Get Verbal Feedback
    verbal_feedback_file = data_src_path + '/' + 'verbal_feedback.csv'
    verbal_feedbacks = read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')

    # eye_tracking_data = read_file(eye_tracking_file, time_format='%H-%M-%S-%f', date_time_required=False)
    # head_tracking_data = read_file(head_tracking_file, time_format='%H-%M-%S-%f', date_time_required=False)

    clips_save_folder = data_save_path + '/clips'

    # Make Sure to create the data save directory
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    if not os.path.exists(clips_save_folder):
        os.makedirs(clips_save_folder)

    # Save The Ground Truth Sickness
    ground_truth_file = data_save_path + '/ground_truth_sickness.csv'
    verbal_feedbacks['Time'] = verbal_feedbacks['Time'].dt.strftime('%H-%M-%S')
    verbal_feedbacks.to_csv(ground_truth_file, index=False)

    for index, verbal_feedback_frames in verbal_feedbacks.iterrows():
        frame_at_time_t = verbal_feedback_frames['Frame']
        verbal_feedback = verbal_feedback_frames['CS']

        # Get the frame and Time of Interest
        matched_frame = search_a_frame(frame_list, frame_at_time_t)
        frame_of_interest, time_of_interest = get_frame_and_time_of_interest(frame_list, frame_at_time_t, matched_frame,
                                                                             window_size=window)
        # # Save Frames
        # frame_save_dir = data_save_dir + '/frames/Frame-' + str(frame_at_time_t) + '_CS-' + str(verbal_feedback) + "/"
        # save_frames(frame_save_dir, frame_of_interest, image_directory=image_dir)

        # Save the video clips
        video_name = clips_save_folder + '/clip-FR-' + str(frame_at_time_t) + '_CS-' + str(verbal_feedback) + '.avi'
        create_and_save_video(video_name, frame_of_interest, image_directory=image_dir)

        # # Save Disparity Map
        # video_name = data_save_dir + '/disparity/disp-FR-' + str(frame_at_time_t) + '_CS-' + str(
        #     verbal_feedback) + '.mp4'
        # create_disparity_video(video_name, frame_of_interest, image_directory=image_dir)

        # # Save Eye_tracking
        # file = data_save_dir + '/eye-tracking/eye-FR-' + str(frame_at_time_t) + '_CS-' + str(verbal_feedback) + '.csv'
        # data = eye_tracking_data[eye_tracking_data['Time'].isin(time_of_interest)]
        # data.to_csv(file, index=False)

        # Save Head Tracking
        # file = data_save_dir + '/head-tracking/head-FR-' + str(frame_at_time_t) + '_CS-' + str(verbal_feedback) + '.csv'
        # data = head_tracking_data[head_tracking_data['Time'].isin(time_of_interest)]
        # data.to_csv(file, index=False)

    return


def main(data_path, data_save_directory, make_class=False):
    simulations = os.listdir(data_path)
    print("Simulation List: ", simulations)
    for simulation in simulations:
        simulation_path = os.path.join(data_path, simulation + '/')
        individual_list = os.listdir(simulation_path)
        for individual in individual_list:
            print(f"Processing Individual {individual} in simulation {simulation}")
            indiv_data_save_dir = os.path.join(data_save_directory, simulation + '/' + individual)

            # Creating data Save Directories
            if not make_class:
                if not os.path.exists(indiv_data_save_dir):
                    os.makedirs(indiv_data_save_dir)

            # Individual Raw Data Path
            individual_raw_data_path = os.path.join(simulation_path, individual + '/')

            if not make_class:
                process_data_per_individual(individual_raw_data_path, indiv_data_save_dir)
            else:
                process_data_per_class(individual_raw_data_path, data_save_directory)


# ................................................. SETUP CONFIGURATIONS ..............................................
if __name__ == "__main__":
    # Video Save Config
    frame_size = (512, 256)
    window = 10
    fps = 20
    path = '../../data/raw/'
    data_save_dir = '../../data/processed/'
    class_rule = {'low': 1, 'medium': 4, 'high': 10}

    main(path, data_save_dir, make_class=True)
# ................................................. SETUP CONFIGURATIONS END ...........................................
