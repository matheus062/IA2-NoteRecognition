import librosa
import json

SAMPLE_RATE = 22050
DURATION = 0.1
NUM_SEGMENTS = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

if __name__ == "__main__":
    print("Pre processing sample...")
    num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    # signal, sr = librosa.load("sample.wav", sr=SAMPLE_RATE)
    signal, sr = librosa.load("output.wav", sr=SAMPLE_RATE)
    data = {
        "mfcc": []
    }

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

    with open("result.json", "w") as fp:
        json.dump(data, fp, indent=4)
