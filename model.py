import keras


def get_model() -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(100, 100, 1)),
        keras.layers.Conv2D(8, 5, activation='relu', kernel_initializer='variance_scaling'),
        keras.layers.MaxPooling2D(strides=(2, 2)),
        keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='variance_scaling'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=10, activation='softmax', kernel_initializer='variance_scaling')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model