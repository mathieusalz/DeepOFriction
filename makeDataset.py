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

    all_rows = []

    for i in range(len(df_velocity)):
        vel_row = df_velocity.iloc[i].values  # shape: (num_timestamps,)
        fric_row = df_friccoef.iloc[i].values  # shape: (num_timestamps,)
        
        for j in range(num_timestamps):
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
