from pathlib import Path


def load_librispeech_dataset(root_dir="LibriSpeech/test-clean", max_samples=50):
    """
    Loads LibriSpeech audio files and corresponding transcripts.

    Returns:
        List of dictionaries:
        {
            "id": str,
            "audio_path": Path,
            "text": str
        }
    """

    root = Path(root_dir)

    samples = []

    # Find every transcript file
    transcript_files = list(root.rglob("*.txt"))

    for transcript_file in transcript_files:

        with open(transcript_file, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split(" ", 1)

                file_id = parts[0]

                transcription = parts[1]

                audio_path = transcript_file.parent / f"{file_id}.flac"

                if audio_path.exists():

                    samples.append(
                        {
                            "id": file_id,
                            "audio_path": audio_path,
                            "text": transcription,
                        }
                    )

                if len(samples) >= max_samples:
                    return samples

    return samples