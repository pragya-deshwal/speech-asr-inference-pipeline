from pathlib import Path

from src.dataset_loader import load_librispeech_dataset
from src.inference import load_model, transcribe_dataset
from src.utils import save_predictions
from src.evaluation import evaluate_predictions
from src.report import generate_report


def main():

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Loading Dataset
    print("=" * 60)
    print("Automatic Speech Recognition Evaluation Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")

    dataset = load_librispeech_dataset(max_samples=50)

    print(f"Loaded {len(dataset)} samples\n")

    # Loading Model
    print("\n[2/5] Loading model...")

    processor, model = load_model()

    # Running Interface
    print("\n[3/5] Running inference...")

    predictions, avg_latency = transcribe_dataset(
        dataset,
        processor,
        model
    )

    # Saving Predictions
    predictions_path = results_dir / "predictions.csv"

    save_predictions(
        predictions,
        predictions_path
    )

    # Evaluation
    print("\n[4/5] Evaluating predictions...")
    metrics_path = results_dir / "metrics.json"

    metrics = evaluate_predictions(
        predictions,
        avg_latency,
        metrics_path
    )

    # Generating Report
    print("\n[5/5] Generating report...")
    report_path = results_dir / "report.md"

    generate_report(
        metrics,
        report_path
    )

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    print(f"\nProcessed Samples : {metrics['processed_samples']}")
    print(f"Average Latency   : {metrics['average_latency_seconds']:.4f} sec")
    print(f"Word Error Rate   : {metrics['word_error_rate']:.4f}")
    print(f"Character Error Rate : {metrics['character_error_rate']:.4f}")

    print("\nGenerated Files:")
    print(f"✓ {predictions_path}")
    print(f"✓ {metrics_path}")
    print(f"✓ {report_path}")

if __name__ == "__main__":
    main()