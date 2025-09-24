import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_loss_curves(csv_path, output_image_path):
    """
    Reads a CSV file with experiment loss data and plots training and validation loss curves.

    Args:
        csv_path (str): The path to the input CSV file.
        output_image_path (str): The path to save the output plot image.
    """
    # Read the data from the CSV file
    df = pd.read_csv(csv_path)

    # Set up the plot style
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define the desired order of experiments
    experiment_order = [
        "bigram",
        "sa",
        "mha",
        "ffn",
        "resconnections",
        "postlayernorm",
        "prelayernorm",
        "scaleup",
    ]
    # Get the unique experiment names to assign different colors
    experiments = df["experiment"].unique()
    colors = plt.cm.get_cmap("tab10", len(experiments))
    color_map = {exp: colors(i) for i, exp in enumerate(experiments)}

    legend_map = {
        "bigram": "Baseline (Bigram)",
        "sa": "Self-Attention (Single Head)",
        "mha": "Multi-Head Attention",
        "ffn": "Feed-Forward Network",
        "resconnections": "Residual Connections",
        "postlayernorm": "Post-Layer Normalization",
        "prelayernorm": "Pre-Layer Normalization",
        "scaleup": "Scaled-up Architecture",
    }

    # Plot Training Loss
    axes[0].set_title("Training Loss vs. Steps")
    for experiment in experiment_order:
        if experiment in experiments:
            exp_df = df[df["experiment"] == experiment]
            label = legend_map.get(experiment, experiment)
            last_loss = exp_df["train_loss"].iloc[-1]
            label = f"{label} ({last_loss:.4f})"
            axes[0].plot(
                exp_df["step"],
                exp_df["train_loss"],
                label=label,
                color=color_map[experiment],
            )

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend()

    # Plot Validation Loss
    axes[1].set_title("Validation Loss vs. Steps")
    for experiment in experiment_order:
        if experiment in experiments:
            exp_df = df[df["experiment"] == experiment]
            label = legend_map.get(experiment, experiment)
            last_loss = exp_df["val_loss"].iloc[-1]
            label = f"{label} ({last_loss:.4f})"
            axes[1].plot(
                exp_df["step"],
                exp_df["val_loss"],
                label=label,
                color=color_map[experiment],
            )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Validation Loss")
    axes[1].legend()

    # # Add a main title for the entire figure
    # fig.suptitle("Training and Validation Loss Curves for All Experiments", fontsize=16)

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot to a file
    plt.savefig(output_image_path)
    print(f"Plot saved to {output_image_path}")

    # Display the plot
    plt.show()


# Example usage with your CSV file:
csv_file_path = "../data/loss_gpt_pd.csv"
output_plot_path = "../images/loss_curves.png"
plot_loss_curves(csv_file_path, output_plot_path)
