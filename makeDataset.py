import pandas as pd
import numpy as np

def createDataset():
    file_VELOCITY = "synthetic_data_generation/features.csv"
    file_FRICCOEF = 'synthetic_data_generation/targets_AgingLaw.csv'

    # Load datasets
    df_velocity = pd.read_csv(file_VELOCITY, header=None)
    df_friccoef = pd.read_csv(file_FRICCOEF, header=None)

    num_timestamps = df_velocity.shape[1]

    # Rename velocity columns
    df_velocity.columns = [f'vel{i+1}' for i in range(num_timestamps)]

    # Time array (must match number of columns/timestamps)
    Vmax = 1.0e-1
    Dc = 1.0e-1
    nTransient = 300
    delta_t = (Dc / Vmax) / nTransient
    total_time = delta_t * num_timestamps
    time = np.arange(0, total_time, delta_t)

    assert df_friccoef.shape[1] == num_timestamps, "Velocity and friction data must have the same number of timestamps"

    # --- Apply normalization and transformation here ---

    # Convert friction coefficient dataframe to numpy array for processing
    fric_coef_np = df_friccoef.to_numpy()

    # Normalize friction coefficient to [0,1]
    max_fric = fric_coef_np.max()
    min_fric = fric_coef_np.min()
    fric_coef_norm = (fric_coef_np - min_fric) / (max_fric - min_fric)

    # Convert velocity dataframe to numpy array for processing
    velocity_np = df_velocity.to_numpy()

    # Apply the double-log transform to velocity, careful with zeros or negatives!
    # Add a small epsilon to avoid division by zero or log of zero

    velocity_log = np.log(np.log(1 / velocity_np))

    # Normalize velocity_log to [0,1]
    min_vel = velocity_log.min()
    max_vel = velocity_log.max()
    velocity_norm = (velocity_log - min_vel) / (max_vel - min_vel)

    # --- End normalization ---

    all_rows = []

    for i in range(len(df_velocity)):
        vel_row = velocity_norm[i]  # normalized velocity row (num_timestamps,)
        fric_row = fric_coef_norm[i]  # normalized friction coefficient row (num_timestamps,)

        for j in range(num_timestamps):
            # Compose each row with all velocity components, timestamp, and friction coefficient at that timestamp
            row = list(vel_row) + [time[j], fric_row[j]]
            all_rows.append(row)

    # Create DataFrame
    col_names = [f'vel{i+1}' for i in range(num_timestamps)] + ['t', 'friction_coefficient']
    df_final = pd.DataFrame(all_rows, columns=col_names)

    # Save to CSV
    output_file = 'synthetic_data_generation/combined_dataset.csv'
    df_final.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}")

if __name__ == "__main__":
    createDataset()
