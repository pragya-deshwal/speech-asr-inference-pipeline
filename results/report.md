# Automatic Speech Recognition Evaluation Report

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

Samples evaluated: 50

---

## Evaluation Results

Word Error Rate (WER): 0.143

Character Error Rate (CER): 0.0342

Average inference latency:

0.4865 seconds

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
