import pandas as pd
import numpy as np

# Your CSV from GATE simulation
df = pd.read_csv(r'C:\Users\h\Desktop\PetStuff\groundtruth.csv')

# Method 1: Extract last 6 columns by position
coordinates = df.iloc[:, -6:].values  # Last 6 columns

# Save as numpy array (uncompressed)
np.save('listmode_coordinates.npy', coordinates)

print(f"\nFiles saved:")
print(f"- listmode_coordinates.npy (float64)")

# Quick test: Load the numpy file to verify
loaded_coords = np.load('listmode_coordinates.npy')
print(f"\nVerification - loaded shape: {loaded_coords.shape}")

# Split into LOR start and end points for pytomography
lor_start = loaded_coords[:, :3]  # First 3 columns (x1, y1, z1)
lor_end = loaded_coords[:, 3:]    # Last 3 columns (x2, y2, z2)

print(f"LOR start points shape: {lor_start.shape}")
print(f"LOR end points shape: {lor_end.shape}")

# Columns are structured :
# - x1, y1, z1, x2, y2, z2