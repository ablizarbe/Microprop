import pandas as pd
import numpy as np


_data = [
    [12.8, 12.7, 12.939, 13.124, 13.466, 13.765, 14.027, 14.250, 14.820, 15.190, 15.479, 15.706, 15.888, 16.047, 16.191, 16.326, 16.451, 16.532],
    [13.476, 13.537, 13.670, 13.796, 14.020, 14.212, 14.376, 14.518, 14.869, 15.105, 15.339, 15.575, 15.815, 16.063, 16.306, 16.541, 16.765, 16.903],
    [14.49, 14.47, 14.500, 14.527, 14.576, 14.616, 14.650, 14.682, 14.83, 15.007, 15.200, 15.416, 15.660, 15.901, 16.301, 16.627, 16.932, 17.391],
    [15.454, 15.305, 15.207, 15.11, 14.969, 14.857, 14.761, 14.704, 14.729, 14.998, 15.374, 15.806, 16.237, 16.645, 17.021, 17.364, 17.675, 17.866],
    [16.335, 16.122, 15.775, 15.557, 15.207, 14.937, 14.748, 14.641, 14.687, 15.366, 15.572, 16.095, 16.591, 17.043, 17.449, 17.811, 18.153, 18.329],
    [17.114, 16.878, 16.194, 15.860, 15.281, 14.895, 14.650, 14.546, 14.71, 15.247, 15.859, 16.455, 16.994, 17.472, 17.891, 18.259, 18.583, 18.779],
    [17.779, 17.442, 16.462, 15.97, 15.236, 14.778, 14.540, 14.467, 14.82, 15.505, 16.215, 16.853, 17.429, 17.917, 18.330, 18.706, 19.023, 19.214],
    [18.318, 17.342, 16.579, 15.969, 15.095, 14.623, 14.434, 14.438, 15.020, 15.845, 16.620, 17.307, 17.883, 18.371, 18.786, 19.143, 19.452, 19.635],
    [18.754, 17.502, 16.554, 15.826, 14.894, 14.471, 14.378, 14.485, 15.336, 16.251, 17.082, 17.77, 18.350, 18.820, 19.231, 19.574, 19.868, 20.043],
    [19.017, 17.50, 16.330, 15.554, 14.571, 14.359, 14.398, 14.525, 15.67, 16.713, 17.571, 18.204, 18.825, 19.286, 19.671, 19.995, 20.273, 20.437],
    [19.13, 17.358, 16.12, 15.292, 14.407, 14.322, 14.519, 14.973, 16.149, 17.229, 18.1, 18.764, 19.304, 19.742, 20.104, 20.408, 20.666, 20.816],
    [19.114, 17.377, 15.7, 14.955, 14.323, 14.372, 14.758, 15.232, 16.454, 17.778, 18.626, 19.275, 19.785, 20.195, 20.530, 20.810, 21.047, 21.186],
    [18.935, 16.9, 15.138, 14.63, 14.283, 14.536, 15.131, 15.711, 17.247, 14.367, 19.181, 19.793, 20.266, 20.641, 20.940, 21.203, 21.417, 21.542],
    [18.532, 16.142, 14.393, 14.372, 14.135, 14.962, 15.643, 16.312, 17.904, 18.706, 19.752, 20.315, 20.746, 21.085, 21.360, 21.587, 21.776, 21.887],
    [18.062, 15.34, 14.510, 14.25, 14.703, 15.515, 16.321, 17.034, 19.41, 19.557, 20.335, 20.841, 21.223, 21.523, 21.763, 21.960, 22.125, 22.221],
    [17.346, 15.929, 14.247, 14.366, 15.266, 16.277, 17.154, 17.879, 19.389, 20.312, 20.929, 21.349, 21.699, 21.959, 22.159, 22.326, 22.463, 22.545],
    [16.411, 14.412, 14.319, 14.829, 16.141, 17.271, 18.155, 18.867, 20.213, 21.012, 21.533, 21.900, 22.172, 22.382, 22.549, 22.684, 22.797, 22.862],
    [15.336, 14.251, 14.892, 15.808, 17.384, 18.514, 19.330, 19.940, 21.090, 21.736, 22.149, 22.436, 22.648, 22.819, 22.939, 23.043, 23.130, 23.180],
    [14.342, 14.971, 16.352, 17.496, 19.058, 20.034, 20.694, 21.169, 22.031, 22.499, 22.795, 23.000, 23.152, 23.269, 23.363, 23.440, np.nan, np.nan],
    [15.055, 17.619, 19.179, 20.153, 21.298, 21.912, 22.351, 22.6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
]

# Row and column labels
_index_labels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
_column_labels = ['5°', '10°', '15°', '20°', '30°', '40°', '50°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°', '350°']

# The DataFrame is created here and can be imported by other files
image_data_table = pd.DataFrame(_data, index=_index_labels, columns=_column_labels)
image_data_table.index.name = 'r'
image_data_table.columns.name = 'a'


def get_Nu(r_value, a_value):
    # --- 1. Parse and Validate Inputs ---
    
    # Get the numeric representations of the table's axes
    r_labels = image_data_table.index.values
    a_numeric_labels = np.array([int(col.replace('°', '')) for col in image_data_table.columns])

    # Convert the input 'a_value' (angle) to a number
    try:
        if isinstance(a_value, str):
            a_input_numeric = float(a_value.replace('°', ''))
        else:
            a_input_numeric = float(a_value)
    except (ValueError, AttributeError):
        return "Error: Invalid format for 'a_value'. It should be a number or a string like '25°'."

    # --- 2. Boundary Checks ---
    if not (r_labels.min() <= r_value <= r_labels.max()):
        return f"Error: r_value '{r_value}' is outside the valid range [{r_labels.min()}, {r_labels.max()}]."
    
    if not (a_numeric_labels.min() <= a_input_numeric <= a_numeric_labels.max()):
        return f"Error: a_value '{a_input_numeric}°' is outside the valid range [{a_numeric_labels.min()}°, {a_numeric_labels.max()}°]."

    # --- 3. Find Bounding Points for Interpolation ---
    
    # Find bounding indices for the r-axis
    r_idx = np.searchsorted(r_labels, r_value)
    if r_idx > 0 and r_labels[r_idx] == r_value: # Exact match
        r1_idx = r2_idx = r_idx
    else:
        r1_idx, r2_idx = r_idx - 1, r_idx
    
    # Find bounding indices for the a-axis
    a_idx = np.searchsorted(a_numeric_labels, a_input_numeric)
    if a_idx > 0 and a_numeric_labels[a_idx] == a_input_numeric: # Exact match
        a1_idx = a2_idx = a_idx
    else:
        a1_idx, a2_idx = a_idx - 1, a_idx
        
    # Get the coordinate values of the four corners surrounding the target point
    r1, r2 = r_labels[r1_idx], r_labels[r2_idx]
    a1, a2 = a_numeric_labels[a1_idx], a_numeric_labels[a2_idx]

    # Retrieve the values at these four corners from the DataFrame
    q11 = image_data_table.iloc[r1_idx, a1_idx]
    q12 = image_data_table.iloc[r1_idx, a2_idx]
    q21 = image_data_table.iloc[r2_idx, a1_idx]
    q22 = image_data_table.iloc[r2_idx, a2_idx]

    # Check for NaN values, which make interpolation impossible
    if pd.isna(q11) or pd.isna(q12) or pd.isna(q21) or pd.isna(q22):
        return "Error: Cannot interpolate because the surrounding data points contain missing values (NaN)."

    # --- 4. Perform Bilinear Interpolation ---

    # If the point lies exactly on a grid line or point, the math simplifies.
    # Case 1: Point is an exact match on an existing grid point
    if r1 == r2 and a1 == a2:
        return q11

    # Case 2: Point is on a horizontal grid line (r is exact)
    if r1 == r2:
        return q11 + (q12 - q11) * (a_input_numeric - a1) / (a2 - a1)

    # Case 3: Point is on a vertical grid line (a is exact)
    if a1 == a2:
        return q11 + (q21 - q11) * (r_value - r1) / (r2 - r1)
        
    # Case 4: General case - requires full bilinear interpolation
    # First, interpolate along the 'a' axis for the top boundary (r1)
    r1_interp = q11 + (q12 - q11) * (a_input_numeric - a1) / (a2 - a1)
    
    # Second, interpolate along the 'a' axis for the bottom boundary (r2)
    r2_interp = q21 + (q22 - q21) * (a_input_numeric - a1) / (a2 - a1)
    
    # Finally, interpolate along the 'r' axis between the two results above
    final_value = r1_interp + (r2_interp - r1_interp) * (r_value - r1) / (r2 - r1)
    
    return final_value