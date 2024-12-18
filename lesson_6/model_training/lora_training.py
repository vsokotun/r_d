import subprocess
import re
import matplotlib.pyplot as plt

def run_and_plot_training():
    command = [
        "mlx_lm.lora",
        "--model", "models/llama3.1",
        "--data", "data",
        "--train",
        "--batch-size", "1",
        "--iters", "600",
        "--steps-per-report", "1",
        "--adapter-path", "adapters",
        "--max-seq-length", "5098",
        "--steps-per-eval", "5",
        "--grad-checkpoint",
        "--learning-rate", "0.0001"
    ]

    train_loss_pattern = re.compile(r"Iter (\d+): Train loss ([0-9.]+)")
    val_loss_pattern = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")

    train_losses = []
    val_losses = []
    iterations = []
    val_iterations = []

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            print(line.strip())

            # Витягуємо трейн-лосси з терміналу
            train_match = train_loss_pattern.search(line)
            if train_match:
                iter_num = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                iterations.append(iter_num)
                train_losses.append(train_loss)

            # Витягуємо валідаційні лосси з терміналу
            val_match = val_loss_pattern.search(line)
            if val_match:
                iter_num = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                val_iterations.append(iter_num)
                val_losses.append(val_loss)

        process.wait()

        if process.returncode != 0:
            print("Training failed! Check the command or parameters.")

    except FileNotFoundError:
        print("Error: Command 'mlx_lm.lora' not found. Ensure mlx is installed and in your PATH.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    plt.figure(figsize=(10, 6))

    if train_losses:
        plt.plot(iterations, train_losses, label="Train Loss", marker='o')

    if val_losses:
        plt.plot(val_iterations, val_losses, label="Validation Loss", marker='x')

    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plot_file = "training_loss_plot.png"
    plt.savefig(plot_file)
    print(f"Loss plot saved as {plot_file}")
    plt.show()

if __name__ == "__main__":
    run_and_plot_training()
