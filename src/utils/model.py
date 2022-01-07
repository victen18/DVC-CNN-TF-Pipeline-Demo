import tensorflow as tf
import logging
import io


def __get_model_summary(model):
    with io.StringIO() as stream:
        model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
        summary_str = stream.getvalue()
        return summary_str


def get_VGG16_model(input_shape: list, model_path: str) -> tf.keras.models.Model:
    """saving and returning the base model extracted from vgg16"""
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    logging.info(f"VGG16 base model summary:\n{__get_model_summary(model)}")
    model.save(model_path)
    logging.info(f"VGG16 model saved at:{model_path}")
    return model


def prepare_full_model(
    base_model,
    learning_rate,
    CLASSES=2,
    freeze_all=True,
    freeze_till=None,
) -> tf.keras.models.Model:

    if freeze_all:
        for layer in base_model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in base_model.layers[:-freeze_till]:
            layer.trainable = False

    # add our layers to base model

    flatten_in = tf.keras.layers.Flatten()(base_model.output)

    prediction = tf.keras.layers.Dense(units=CLASSES, activation="softmax")(flatten_in)

    full_model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)

    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    logging.info("custom model is compiled and ready to be trained")
    logging.info(f"full model summary:{__get_model_summary(full_model)}")

    return full_model
