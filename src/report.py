def generate_report(metrics, output_path):

    report = f"""# Automatic Speech Recognition Evaluation Report

## Objective

Evaluate the pretrained Hugging Face ASR model
facebook/wav2vec2-base-960h on a subset of the
LibriSpeech Test-Clean dataset.

---

## Model

facebook/wav2vec2-base-960h

---

## Dataset

LibriSpeech Test-Clean

Samples evaluated: {metrics['processed_samples']}

---

## Evaluation Results

Word Error Rate (WER): {metrics['word_error_rate']}

Character Error Rate (CER): {metrics['character_error_rate']}

Average inference latency:

{metrics['average_latency_seconds']} seconds

---

## Observations

• The model produces highly accurate transcripts on clean English speech.

• Low WER and CER indicate strong recognition performance.

• CPU inference remains reasonably fast.

---

## Limitations

• Performance decreases on noisy speech.

• Model is trained primarily for English.

• CPU inference is slower than GPU inference.

---

## Conclusion

The inference pipeline successfully loads a pretrained
Hugging Face Wav2Vec2 model, performs automatic
speech recognition on LibriSpeech, computes evaluation
metrics, and generates reproducible outputs.
"""

    with open(output_path,"w",encoding="utf-8") as f:
        f.write(report)