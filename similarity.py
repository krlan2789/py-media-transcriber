import argparse
import json


def levenshtein_distance(a, b):
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # Deletion
                matrix[i][j - 1] + 1,  # Insertion
                matrix[i - 1][j - 1] + cost,
            )  # Substitution

    return matrix[len(a)][len(b)]


def similarity_percentage(a, b):
    distance = levenshtein_distance(a, b)
    max_length = max(len(a), len(b))
    return ((max_length - distance) / max_length) * 100


parser = argparse.ArgumentParser()
parser.add_argument("-t", help="Transcription results")
parser.add_argument("-s", help="Original sample text")
args = parser.parse_args()

transcription = args.t
originalSample = args.s

results = {
    "status": False,
}

if __name__ == "__main__":
    if transcription and originalSample:
        results = {
            "status": True,
            "similarity": f"{similarity_percentage(transcription, originalSample):.1f}%",
            "transcription": transcription,
            "original": originalSample,
        }

print(json.dumps(results, ensure_ascii=False))
