import os
import json
import numpy as np
import matplotlib.pyplot as plt


def resolve_path(path):
    """Resolve paths relative to the project root or current directory."""
    if os.path.exists(path):
        return path

    alt_path = os.path.join("../../", path)

    if os.path.exists(alt_path):
        return alt_path

    return path


def load_json(path):
    """Load JSON data and report errors clearly."""
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"[ERROR] Cannot load {path}: {error}")
        return None


def extract_histories(data, metric_name):
    """Extract valid, non-empty numerical histories."""
    if data is None:
        return []

    runs = data if isinstance(data, list) else [data]
    histories = []

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            continue

        values = run.get(metric_name)

        if not isinstance(values, list) or not values:
            continue

        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            print(
                f"[WARNING] Invalid {metric_name} "
                f"in run {index}."
            )
            continue

        if values.ndim != 1:
            print(
                f"[WARNING] {metric_name} must be one-dimensional."
            )
            continue

        if not np.all(np.isfinite(values)):
            print(
                f"[WARNING] Non-finite values found in "
                f"{metric_name}, run {index}. Skipping run."
            )
            continue

        histories.append(values)

    return histories


def plot_real_metric(ax, histories, label):
    """Plot actual observations without generating synthetic values."""
    if not histories:
        print(
            f"[WARNING] No genuine recorded data for {label}. "
            "The curve will not be plotted."
        )
        return

    if len(histories) == 1:
        values = histories[0]
        ax.plot(
            np.arange(len(values)),
            values,
            label=label,
            linewidth=1.2
        )
        print(f"[INFO] {label}: {len(values)} real observations.")
        return

    lengths = [len(history) for history in histories]

    if len(set(lengths)) != 1:
        print(
            f"[WARNING] Cannot average {label}: "
            f"history lengths differ: {lengths}"
        )

        # Plot each run separately rather than silently truncating.
        for index, values in enumerate(histories):
            ax.plot(
                np.arange(len(values)),
                values,
                linewidth=1.0,
                label=f"{label} run {index + 1}"
            )
        return

    values = np.stack(histories, axis=0)
    mean_values = np.mean(values, axis=0)

    ax.plot(
        np.arange(len(mean_values)),
        mean_values,
        label=f"{label} mean",
        linewidth=1.2
    )

    if len(histories) > 1:
        std_values = np.std(values, axis=0, ddof=1)
        ax.fill_between(
            np.arange(len(mean_values)),
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
            label=f"{label} ± 1 SD"
        )

    print(
        f"[INFO] {label}: {len(mean_values)} observations, "
        f"{len(histories)} runs."
    )


def plot_graph(
    train_summary_path="checkpoints/train_summary_full.json",
    test_eval_path="checkpoints/test_evaluation_summary.json",
    output_dir="checkpoints/visualizations"
):
    train_path = resolve_path(train_summary_path)
    test_path = resolve_path(test_eval_path)
    output_dir = resolve_path(output_dir)

    train_data = load_json(train_path)

    if train_data is None:
        print("[ERROR] Training data unavailable. Aborting.")
        return

    test_data = load_json(test_path)

    os.makedirs(output_dir, exist_ok=True)

    # Only plot metrics genuinely recorded in the training summary.
    metrics = {
        "loss_history": "Training Loss",
        "validation_loss_history": "Validation Loss",
        "reward_history": "Training Reward",
        "agreement_history": "Agent Agreement"
    }

    for metric_name, title in metrics.items():
        histories = extract_histories(train_data, metric_name)

        if not histories:
            print(
                f"[INFO] {metric_name} is not available "
                "as a valid recorded history."
            )
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        plot_real_metric(ax, histories, title)

        ax.set_title(title)
        ax.set_xlabel("Recorded Step / Episode")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.25)

        filename = os.path.join(
            output_dir,
            f"{metric_name.replace('_history', '')}.png"
        )

        fig.tight_layout()
        fig.savefig(filename, dpi=300, facecolor="white")
        plt.close(fig)

        print(f"[SUCCESS] Saved: {filename}")

    # Display genuine final test metrics separately.
    if isinstance(test_data, dict):
        print("\n--- Recorded Test Evaluation Metrics ---")

        for metric_name, value in test_data.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                print(f"{metric_name}: {value}")


if __name__ == "__main__":
    plot_graph()