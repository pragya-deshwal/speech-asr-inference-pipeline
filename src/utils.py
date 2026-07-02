import pandas as pd


def save_predictions(predictions, output_file):

    df = pd.DataFrame(predictions)

    columns = [
    "audio_id",
    "ground_truth",
    "prediction",
    "latency"
    ]

    df = df[columns]

    df.to_csv(output_file, index=False)

    print(f"\nSaved predictions to {output_file}")