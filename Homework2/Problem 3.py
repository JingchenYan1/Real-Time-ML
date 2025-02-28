import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU for training: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("Using CPU for training.")

def residual_block(inputs, filters, stride=1, use_projection=False, l2_reg=1e-4):
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    # If dimensions do not match, use projection shortcut
    if use_projection:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False,
                                 kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet11(input_shape, num_classes, l2_reg=1e-4):
    """
    Construct ResNet-11 model:
      - Initial 3x3 convolution (64 filters)
      - Group 1: 2 residual blocks (64 filters)
      - Group 2: 1 residual block (128 filters, stride 2, with projection)
      - Group 3: 1 residual block (256 filters, stride 2, with projection)
      - Group 4: 1 residual block (512 filters, stride 2, with projection)
      - Global average pooling + fully connected classifier
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Group 1: Two residual blocks, 64 filters
    x = residual_block(x, 64, stride=1, use_projection=False, l2_reg=l2_reg)
    x = residual_block(x, 64, stride=1, use_projection=False, l2_reg=l2_reg)

    # Group 2: 1 residual block, 128 filters, stride 2 (with projection)
    x = residual_block(x, 128, stride=2, use_projection=True, l2_reg=l2_reg)

    # Group 3: 1 residual block, 256 filters, stride 2 (with projection)
    x = residual_block(x, 256, stride=2, use_projection=True, l2_reg=l2_reg)

    # Group 4: 1 residual block, 512 filters, stride 2 (with projection)
    x = residual_block(x, 512, stride=2, use_projection=True, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def build_resnet18(input_shape, num_classes, l2_reg=1e-4):
    """
    Construct ResNet-18 model:
      - Initial 3x3 convolution (64 filters)
      - Group 1: 2 residual blocks (64 filters)
      - Group 2: 2 residual blocks (128 filters, first block stride 2, with projection)
      - Group 3: 2 residual blocks (256 filters, first block stride 2, with projection)
      - Group 4: 2 residual blocks (512 filters, first block stride 2, with projection)
      - Global average pooling + fully connected classifier
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Group 1: Two residual blocks, 64 filters
    x = residual_block(x, 64, stride=1, use_projection=False, l2_reg=l2_reg)
    x = residual_block(x, 64, stride=1, use_projection=False, l2_reg=l2_reg)

    # Group 2: Two residual blocks, 128 filters, first block stride 2
    x = residual_block(x, 128, stride=2, use_projection=True, l2_reg=l2_reg)
    x = residual_block(x, 128, stride=1, use_projection=False, l2_reg=l2_reg)

    # Group 3: Two residual blocks, 256 filters, first block stride 2
    x = residual_block(x, 256, stride=2, use_projection=True, l2_reg=l2_reg)
    x = residual_block(x, 256, stride=1, use_projection=False, l2_reg=l2_reg)

    # Group 4: Two residual blocks, 512 filters, first block stride 2
    x = residual_block(x, 512, stride=2, use_projection=True, l2_reg=l2_reg)
    x = residual_block(x, 512, stride=1, use_projection=False, l2_reg=l2_reg)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def plot_history(history, title):
    """Plot training and validation loss and accuracy curves."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title + " - Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title + " - Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def train_and_evaluate(model_builder, dataset_name, model_name, epochs=30, batch_size=64):
    """
    Construct the model based on the provided model_builder, train it on the specified dataset, and plot the training curve.
    Added label smoothing loss and ReduceLROnPlateau callback (only for learning rate decay, no early stopping).
    """
    # Load dataset
    if dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError("Dataset must be 'cifar10' or 'cifar100'")

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Build model
    model = model_builder(input_shape=(32, 32, 3), num_classes=num_classes)

    # Use label smoothing cross-entropy loss
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print(f"\n===== Training {model_name} on {dataset_name.upper()} =====")
    model.summary()
    print("Total model parameters:", model.count_params())

    # Reduce learning rate when validation loss does not improve
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=3, verbose=1, min_lr=1e-6)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_test, y_test), callbacks=[reduce_lr],
                        verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Loss: {loss:.4f}, Final Test Accuracy: {acc:.4f}")
    plot_history(history, f"{model_name} on {dataset_name.upper()}")

    return model, history

print("Training ResNet-11 on CIFAR-10:")
model_resnet11_c10, history_resnet11_c10 = train_and_evaluate(build_resnet11, 'cifar10', 'ResNet-11')

print("\nTraining ResNet-18 on CIFAR-10:")
model_resnet18_c10, history_resnet18_c10 = train_and_evaluate(build_resnet18, 'cifar10', 'ResNet-18')

print("\nTraining ResNet-11 on CIFAR-100:")
model_resnet11_c100, history_resnet11_c100 = train_and_evaluate(build_resnet11, 'cifar100', 'ResNet-11')

print("\nTraining ResNet-18 on CIFAR-100:")
model_resnet18_c100, history_resnet18_c100 = train_and_evaluate(build_resnet18, 'cifar100', 'ResNet-18')