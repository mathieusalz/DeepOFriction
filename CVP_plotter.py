import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

# define custom class
class CustomValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables
        t = invar["t"][:, 0].reshape(6,-1)  # assuming shape (N,1)

        # Number of trajectories/curves to plot (6)
        n_curves = 6

        # Extract velocities for each of the 6 trajectories and timestamps
        # Assuming invar keys besides 't' correspond to velocity components at different timestamps
        # We'll store velocities as velocities[curve_index, timestamp_index]
        velocities = np.zeros((n_curves, 250))

        # Extract velocity data from invar
        velocity_keys = [key for key in invar.keys() if key != "t"]
        for i in range(n_curves):
            for j, key in enumerate(velocity_keys):
                # invar[key] shape is (batch, 1)
                velocities[i, j] = invar[key][i][0]

        # Extract true and predicted friction coefficient for each curve
        fric_true = true_outvar["friction_coefficient"][:, 0].reshape(n_curves, -1)
        fric_pred = pred_outvar["friction_coefficient"][:, 0].reshape(n_curves, -1)

        # Create figure and axes: 2 rows, 3 cols
        fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=100)
        axes = axes.flatten()

        for i in range(n_curves):
            ax = axes[i]

            # Plot true friction coefficient
            ax.plot(t[i], fric_true[i], label="True Friction", color="blue", linewidth=2)

            # Plot predicted friction coefficient
            ax.plot(t[i], fric_pred[i], label="Predicted Friction", color="orange", linestyle="--", linewidth=2)

            # Plot velocity on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(t[i], velocities[i], label="Velocity", color="green", alpha=0.6)
            ax2.set_ylabel("Velocity")

            ax.set_xlabel("Time")
            ax.set_ylabel("Friction Coefficient")
            ax.set_title(f"Trajectory {i+1}")

            # Combine legends from both y axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.tight_layout()

        return [(fig, "custom_plot"),]
