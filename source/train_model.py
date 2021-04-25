from sklearn.model_selection import train_test_split
from source import neural as nn
from util import data_util


def get_data(dataset_directory, classes_list, max_images_per_class):
    features, labels = data_util.create_dataset(classes_list=classes_list, dataset_directory=dataset_directory,
                                                max_images_per_class=max_images_per_class)
    # one_hot_encoded_labels = to_categorical(labels)

    # TODO: Multimodal train test split.
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                test_size=0.2, shuffle=True,
                                                                                random_state=15)
    return features_train, features_test, labels_train, labels_test


def train_model(clip_X_train, optic_X_train, disp_X_train, target, classes_list):
    # Create the model
    input_shape = (128, 128, 3)
    input_clip, flatten_clips = nn.DeepVDs.deep_vds_from_clip(input_shape)
    input_optic, flatten_optics = nn.DeepVDs.deep_vds_from_optic(input_shape)
    input_disp, flatten_disp = nn.DeepVDs.deep_vds_from_dsip(input_shape)

    model = nn.DeepVDs.get_full_model([flatten_clips, flatten_optics, flatten_disp],
                                   [input_clip, input_optic, input_disp], len(classes_list))

    # Adding loss, optimizer and metrics values to the model.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])


    # Start Training
    model_training_history = model.fit(x=[clip_X_train, optic_X_train, disp_X_train], y=target, epochs=300,
                                       batch_size=8, shuffle=False,
                                       validation_split=0.2)

    return model, model_training_history


def evaluate_model(model, X_test, Y_test):
    model_evaluation_history = model.evaluate(X_test, Y_test)
    return model_evaluation_history
