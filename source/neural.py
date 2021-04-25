from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class DeepVDs:
    @staticmethod
    def deep_vds_from_clip(input_shape):
        # We will use a Sequential model for model construction
        input_layer = Input(shape=input_shape)
        X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
        X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(X)
        X = BatchNormalization()(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)
        X = GlobalAveragePooling2D()(X)
        X = Dropout(0.2)(X)
        flatten_layer1 = Flatten()(X)

        return input_layer, flatten_layer1

    @staticmethod
    def deep_vds_from_optic(input_shape):
        input_2 = Input(shape=input_shape)
        X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_2)
        X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(X)
        X = BatchNormalization()(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)
        X = GlobalAveragePooling2D()(X)
        X = Dropout(0.3)(X)
        flatten2 = Flatten()(X)

        return input_2, flatten2

    @staticmethod
    def deep_vds_from_dsip(input_shape):
        input_2 = Input(shape=input_shape)
        X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_2)
        X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(X)
        X = BatchNormalization()(X)
        X = MaxPooling2D(pool_size=(2, 2))(X)
        X = GlobalAveragePooling2D()(X)
        X = Dropout(0.25)(X)
        flatten2 = Flatten()(X)
        return input_2, flatten2

    @staticmethod
    def get_full_model(layer_list, input_list, output_shape):
        merge_layer = concatenate(layer_list)
        X = Dense(256, activation='relu')(merge_layer)
        X = Dropout(0.4)(X)
        X = BatchNormalization()(X)
        output = Dense(output_shape, activation='softmax')(X)
        model = Model(inputs=input_list, outputs=output)
        print(model.summary())
        return model
