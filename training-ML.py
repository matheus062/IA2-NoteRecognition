import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "./dataset/dataset.json"
EXPORT_PATH = "./model/"
EPOCHS = 150


def plot_history(to_plot):
    fig, axs = plt.subplots(2)

    axs[0].plot(to_plot.history["accuracy"], label="Treino (acurácia)")
    axs[0].plot(to_plot.history["val_accuracy"], label="Teste (acurácia)")
    axs[0].set_ylabel("Acurácia")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Acurácia")

    axs[1].plot(to_plot.history["loss"], label="Treino (erro)")
    axs[1].plot(to_plot.history["val_loss"], label="Teste (erro)")
    axs[1].set_ylabel("Erro")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Erro")

    plt.show()


if __name__ == "__main__":
    print("Carregando dataset...")

    with open(DATASET_PATH, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    print("Dataset carregado com sucesso.")

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1, 40)),
        keras.layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(6, activation="softmax"),
    ])

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimiser,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs,
        targets,
        test_size=0.3
    )

    history = model.fit(
        inputs_train,
        targets_train,
        validation_data=(inputs_test, targets_test),
        batch_size=32,
        epochs=EPOCHS
    )

    plot_history(history)

    model.save(filepath=EXPORT_PATH)
