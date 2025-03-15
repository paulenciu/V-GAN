from matplotlib import pyplot as plt


class GradiantPlotter:

    def __init__(self, vgan):
        self.vgan = vgan

    def log(self, *args):
        """
        Plots the gradient norms of the generator over training epochs.

        Args:
            vgan_instance (VGAN): An instance of the VGAN class after training.
        """
        if "generator_gradient_norms" not in self.vgan.train_history:
            raise ValueError(
                "Gradient norms were not computed during training. Modify the `fit` method to compute them.")

        gradient_norms = self.vgan.train_history["generator_gradient_norms"]
        epochs = range(1, len(gradient_norms) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, gradient_norms, marker='o', linestyle='-', color='b')
        plt.title("Generator Gradient Norms Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm (L2)")
        plt.grid(True)
        plt.show()