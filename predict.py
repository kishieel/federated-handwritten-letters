import argparse
import keras
import numpy as np
import tensorflow as tf


class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def main(model_path: str, image_path: str) -> None:
    model = keras.models.load_model(model_path)

    image = keras.utils.load_img(image_path, target_size=(100, 100), color_mode="grayscale")
    image = keras.utils.img_to_array(image)
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)
    score = tf.nn.softmax(prediction[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validator')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)
    args = parser.parse_args()

    main(args.model_path, args.image_path)
