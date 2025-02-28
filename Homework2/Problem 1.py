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

def build_simplified_alexnet(input_shape, num_classes, use_dropout=False, l2_reg=1e-4):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg),
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_reg)))
    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=regularizers.l2(l2_reg)))

    return model

def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

epochs = 30
batch_size = 64
input_shape = (32, 32, 3)

def train_and_evaluate(dataset_name='cifar10', use_dropout=False):
    if dataset_name.lower() == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        num_classes = 100
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = build_simplified_alexnet(input_shape, num_classes, use_dropout=use_dropout, l2_reg=1e-4)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print("Model Summary:")
    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(x_test, y_test), callbacks=[reduce_lr],
                        verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Loss: {loss:.4f}, Final Test Accuracy: {acc:.4f}")

    title = f"{dataset_name.upper()} - {'Dropout' if use_dropout else 'Baseline'} with Label Smoothing & LR Decay"
    plot_history(history, title)

    return model, history

print("Training on CIFAR-10 (Baseline, No Dropout):")
model_cifar10_baseline, history_cifar10_baseline = train_and_evaluate('cifar10', use_dropout=False)

print("\nTraining on CIFAR-10 (With Dropout):")
model_cifar10_dropout, history_cifar10_dropout = train_and_evaluate('cifar10', use_dropout=True)

print("\nTraining on CIFAR-100 (Baseline, No Dropout):")
model_cifar100_baseline, history_cifar100_baseline = train_and_evaluate('cifar100', use_dropout=False)

print("\nTraining on CIFAR-100 (With Dropout):")
model_cifar100_dropout, history_cifar100_dropout = train_and_evaluate('cifar100', use_dropout=True)
