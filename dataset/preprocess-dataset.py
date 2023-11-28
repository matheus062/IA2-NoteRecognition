import os
import librosa
import json

DATASET_PATH = "./audio/"
JSON_PATH = "./dataset.json"
SAMPLE_RATE = 22050
DURATION = 0.1
NUM_SEGMENTS = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path):
    data = {
        "keys": [],
        "mfcc": [],
        "labels": []
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            for s in range(NUM_SEGMENTS):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment
                mfcc = librosa.feature.mfcc(
                    y=signal[start_sample:finish_sample],
                    sr=sr,
                    n_fft=2048,
                    n_mfcc=40,
                    hop_length=512
                )
                mfcc = mfcc.T
                data["mfcc"].append(mfcc.tolist())
                data["labels"].append(i - 1)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
