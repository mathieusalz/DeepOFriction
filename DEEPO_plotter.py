import numpy as np
import matplotlib.pyplot as plt
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

class CustomValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        t = invar["t"][:, 0].reshape(25, -1)
        n_curves = 25

        velocities = np.zeros((25, 250))
        velocity_keys = [key for key in invar.keys() if key != "t"]

        for i in range(n_curves):
            for j, key in enumerate(velocity_keys):
                velocities[i][j] = invar[key][i*250][0]

        fric_true = true_outvar["friction_coefficient"][:, 0].reshape(n_curves, -1)
        fric_pred = pred_outvar["friction_coefficient"][:, 0].reshape(n_curves, -1)

        fig, axes = plt.subplots(5, 5, figsize=(18, 15), dpi=100)
        axes = axes.flatten()

        # For shared legend
        shared_handles = []
        shared_labels = []

        for i in range(n_curves):
            ax = axes[i]

            l1, = ax.plot(t[i], fric_true[i], label="True Friction", color="blue", linewidth=2)
            l2, = ax.plot(t[i], fric_pred[i], label="Predicted Friction", color="orange", linestyle="--", linewidth=2)

            ax2 = ax.twinx()
            l3, = ax2.plot(t[i], velocities[i], label="Velocity", color="green", alpha=0.6)
            ax2.set_ylabel("Velocity")

            ax.set_xlabel("Time")
            ax.set_ylabel("Friction Coefficient")
            ax.set_title(f"Trajectory {i+1}")

            # Only collect labels from first plot
            if i == 0:
                shared_handles.extend([l1, l2, l3])
                shared_labels.extend([l.get_label() for l in [l1, l2, l3]])

        # Adjust layout and add shared legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
        fig.legend(shared_handles, shared_labels, loc='center left', bbox_to_anchor=(0.87, 0.5), fontsize=12)

        return [(fig, "custom_plot"),]
