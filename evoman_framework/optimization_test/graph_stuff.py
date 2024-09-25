
import matplotlib.pyplot as plt
import pandas as pd


# Read the file
# Read the file
file_path = 'results.txt'
data = pd.read_csv(file_path, sep='\s+', header=0)

# Replace non-numeric values in the third column with zero
data.iloc[:, 2] = pd.to_numeric(data.iloc[:, 2], errors='coerce').fillna(0)
data.iloc[:, 1] = pd.to_numeric(data.iloc[:, 1], errors='coerce').fillna(0)
# Plot second and third columns vs first column
plt.plot(data.iloc[:, 0], data.iloc[:, 1], marker='o', label='Second Column')
plt.plot(data.iloc[:, 0], data.iloc[:, 2], marker='x', label='Third Column')

plt.xlabel('First Column')
plt.ylabel('Values')
plt.title('Second and Third Columns vs First Column')
plt.legend()
plt.show()