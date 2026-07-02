# Speech Recognition Inference & Evaluation Pipeline

## Overview

This project implements a complete Automatic Speech Recognition (ASR) inference and evaluation pipeline using a pretrained speech recognition model from Hugging Face. The pipeline loads a publicly available speech dataset, performs transcription using a pretrained Wav2Vec2 model, evaluates the predictions using standard ASR metrics, and generates reproducible outputs.

The project was developed as part of an ML Internship assignment to demonstrate understanding of speech recognition models, Hugging Face integration, inference pipelines, and evaluation methodologies.

---

## Model

**Model:** `facebook/wav2vec2-base-960h`

The model is a pretrained Connectionist Temporal Classification (CTC) based Automatic Speech Recognition model available on Hugging Face.

---

## Dataset

**Dataset:** LibriSpeech Test-Clean

The LibriSpeech dataset is a publicly available English speech corpus containing high-quality read speech derived from audiobooks.

For this project, 50 audio samples from the Test-Clean split were used for inference and evaluation.

---

## Project Structure

```
speech-asr-assignment/
│
├── src/
│   ├── dataset_loader.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── report.py
│   └── utils.py
│
├── results/
│   ├── predictions.csv
│   ├── metrics.json
│   └── evaluation_report.md
│
├── research/
│
├── LibriSpeech/
│
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

* Loads the pretrained Wav2Vec2 model from Hugging Face.
* Loads the LibriSpeech Test-Clean dataset.
* Performs automatic speech recognition on 50 audio samples.
* Measures average inference latency.
* Computes Word Error Rate (WER).
* Computes Character Error Rate (CER).
* Saves predictions to CSV.
* Generates evaluation metrics in JSON format.
* Produces an evaluation report automatically.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the complete pipeline:

```bash
python run.py
```

---

## Output Files

Running the project automatically generates:

### predictions.csv

Contains:

* Audio ID
* Ground Truth Transcript
* Model Prediction
* Inference Latency

### metrics.json

Contains:

* Model Name
* Dataset
* Number of Samples
* Average Inference Latency
* Word Error Rate (WER)
* Character Error Rate (CER)

### evaluation_report.md

Contains a human-readable summary of the evaluation.

---

## Evaluation Metrics

The following evaluation metrics are computed:

* Word Error Rate (WER)
* Character Error Rate (CER)
* Average Inference Latency
* Total Processed Samples

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Torchaudio
* SoundFile
* JiWER
* Pandas
* NumPy

---

## Future Improvements

* GPU acceleration
* Batch inference
* Noise robustness evaluation
* Fine-tuning on domain-specific speech
* Support for multilingual speech recognition

---

## Author

Pragya Deshwal

AI/ML Internship Assignment
