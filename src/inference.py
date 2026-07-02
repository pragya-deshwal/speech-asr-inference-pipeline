import time
from tqdm import tqdm

import torch
import soundfile as sf

from transformers import AutoProcessor
from transformers import AutoModelForCTC


MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_model():
    """
        Load the pretrained Wav2Vec2 model and processor from Hugging Face.

        Returns:
            tuple:
                processor (AutoProcessor)
                model (AutoModelForCTC)
    """

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForCTC.from_pretrained(MODEL_NAME)

    model.eval()

    return processor, model


def transcribe_dataset(dataset, processor, model):
    """
    Run speech-to-text inference on the dataset.

    Args:
        dataset (list): List of LibriSpeech samples.
        processor: Hugging Face processor.
        model: Pretrained Wav2Vec2 model.

    Returns:
        tuple:
            predictions (list)
            average_latency (float)
    """

    predictions = []

    total_latency = 0

    for sample in tqdm(dataset, desc="Running inference"):

        waveform, sample_rate = sf.read(sample["audio_path"])

# Convert stereo to mono if needed
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        inputs = processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        start = time.perf_counter()

        with torch.no_grad():

            logits = model(inputs.input_values).logits

        end = time.perf_counter()

        latency = end - start

        total_latency += latency

        predicted_ids = torch.argmax(logits, dim=-1)

        prediction = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        prediction = prediction.strip().upper()
        ground_truth = sample["text"].strip().upper()

        predictions.append(
        {
            "audio_id": sample["id"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "latency": round(latency, 4)
        }
    )

    average_latency = total_latency / len(predictions)

    return predictions, average_latency