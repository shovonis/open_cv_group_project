import tensorflow as tf
from sklearn.metrics import confusion_matrix
from util import plot_results
from source import train_model
import numpy as np


def main():
    dataset_directory = "data/cs/clips"
    clip_X_train, clip_X_test, clip_Y_train, clip_Y_test = train_model.get_data(dataset_directory=dataset_directory,
                                                                                classes_list=classes_list,
                                                                                max_images_per_class=max_sample)
    model, history = train_model.train_model(clip_X_train, clip_X_train, clip_X_train, clip_Y_train,
                                             classes_list=classes_list)
    # Save the model
    model.save('model/trained_model.h5')

    # Plot training history
    plot_results.plot_train_loss_history(history)
    plot_results.plot_train_acc_history(history)
    # Predict
    predicted_cs = model.predict([clip_X_test, clip_X_test, clip_X_test])
    print("Predicted: ", np.argmax(predicted_cs, axis=1))
    print("Actual: ", clip_Y_test)
    cm = confusion_matrix(clip_Y_test, np.argmax(predicted_cs, axis=1))
    plot_results.plot_confusion_matrix(cm, list(range(2)))
    # Model Evaluate
    print("Evaluating Model: ")
    train_model.evaluate_model(model, X_test=[clip_X_test, clip_X_test, clip_X_test], Y_test=clip_Y_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Reset the Seed
    seed_constant = 15
    np.random.seed(seed_constant)
    tf.random.set_seed(seed_constant)
    image_height, image_width = 128, 128
    max_sample = 5000
    classes_list = ["high", "medium", "low"]
    model_output_size = len(classes_list)
    main()
