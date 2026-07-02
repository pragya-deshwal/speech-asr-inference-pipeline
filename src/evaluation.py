import json
from jiwer import wer, cer


def evaluate_predictions(predictions, avg_latency, output_path):
    references = [p["ground_truth"] for p in predictions]
    hypotheses = [p["prediction"] for p in predictions]

    metrics = {
        "model": "facebook/wav2vec2-base-960h",
        "dataset": "LibriSpeech Test-Clean",
        "processed_samples": len(predictions),
        "average_latency_seconds": round(avg_latency, 4),
        "word_error_rate": round(wer(references, hypotheses), 4),
        "character_error_rate": round(cer(references, hypotheses), 4),
    }

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics