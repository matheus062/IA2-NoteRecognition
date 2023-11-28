import numpy
import json
import keras

EXPORT_PATH = "./model"
NUM_SEGMENTS = 10
LABELS_COUNT = 6

if __name__ == "__main__":
    with open("analyze/result.json", "r") as fp:
        data = json.load(fp)

    inputs = numpy.array(data["mfcc"])

    model = keras.models.load_model(EXPORT_PATH)
    result = model.predict(inputs)

    finalResult = []

    for label in range(LABELS_COUNT):
        a = 0

        for segment in range(NUM_SEGMENTS):
            a += result[segment][label]

        finalResult.append(a / NUM_SEGMENTS)

    print("Probabilidade de C4: " + str(finalResult[0] * 100))
    print("Probabilidade de C5: " + str(finalResult[1] * 100))
    print("Probabilidade de D4: " + str(finalResult[2] * 100))
    print("Probabilidade de D5: " + str(finalResult[3] * 100))
    print("Probabilidade de E4: " + str(finalResult[4] * 100))
    print("Probabilidade de E5: " + str(finalResult[5] * 100))
