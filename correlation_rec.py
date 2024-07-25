import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_paths: list, prefixes: list) -> pd.DataFrame:
    """
    Load and merge data from multiple CSV files.

    Parameters:
    file_paths (list): List of file paths to CSV files.
    prefixes (list): List of prefixes to add to column names.

    Returns:
    pd.DataFrame: Merged DataFrame with prefixed column names.
    """
    data = pd.concat([pd.read_csv(file_path).add_prefix(prefix + '_') for file_path, prefix in zip(file_paths, prefixes)], axis=1)
    return data

def weigh_player_importance(data: pd.DataFrame, prefixes: list) -> np.ndarray:
    """
    Weigh player importance based on games played.

    Parameters:
    data (pd.DataFrame): DataFrame with player data.
    prefixes (list): List of prefixes.

    Returns:
    np.ndarray: Array of weights based on games played.
    """
    g_cols = [prefix + '_G' for prefix in prefixes]
    g_values = data[g_cols].min(axis=1)  # Use the minimum games played across all prefixes
    return np.where(g_values <= 10, 0.5, 1)

def calculate_correlations(data: pd.DataFrame, correlation_target: str, filter_columns: list) -> pd.Series:
    """
    Calculate correlations between columns and a target column.

    Parameters:
    data (pd.DataFrame): DataFrame with player data.
    correlation_target (str): Target column for correlations.
    filter_columns (list): List of columns to filter out.

    Returns:
    pd.Series: Series of correlations with the target column.
    """
    numeric_data = data.drop(filter_columns, axis=1).select_dtypes(include=['int64', 'float64'])
    correlations = numeric_data.apply(lambda x: x * weigh_player_importance(data, prefixes)).corrwith(data[correlation_target])
    return correlations

# Define file paths and prefixes
file_paths = [
    'C:/Users/PC/Desktop/code/data_files/combined/fp_advancedreceiving.csv',
    'C:/Users/PC/Desktop/code/data_files/combined/fp_receiving.csv',
    'C:/Users/PC/Desktop/code/data_files/combined/fp_redzone.csv',
    'C:/Users/PC/Desktop/code/data_files/combined/fp_tenzone.csv'
]
prefixes = ['advancedreceiving', 'receiving', 'redzone', 'tenzone']

# Load and merge data
data = load_data(file_paths, prefixes)

# Handle duplicate column names
data.columns = [col + str(i) if col in data.columns[:i] else col for i, col in enumerate(data.columns)]

# Define correlation target and filter columns
correlation_target = 'receiving_TD'
filter_columns = [col for col in [prefix + '_' + c for prefix in prefixes for c in ['POS', 'G', 'TM', 'Player']] if col in data.columns]

# Calculate correlations for all positions, WR, and TE
correlations_all = calculate_correlations(data, correlation_target, filter_columns)

pos_columns = [prefix + '_POS' for prefix in prefixes]
correlations_wr = calculate_correlations(data[(data[pos_columns] == 'WR').any(axis=1)], correlation_target, filter_columns)
correlations_te = calculate_correlations(data[(data[pos_columns] == 'TE').any(axis=1)], correlation_target, filter_columns)

breaker = '*****************************************************************************'

# Print strong correlations ( absolute value > 0.5 )
print("Strong Correlations (All Positions):")
print(correlations_all[abs(correlations_all) > 0.5].drop('receiving_TD').sort_values(ascending=False))

print(breaker)

print("Strong Correlations (WR):")
print(correlations_wr[abs(correlations_wr) > 0.5].drop('receiving_TD').sort_values(ascending=False))

print(breaker)

print("Strong Correlations (TE):")
print(correlations_te[abs(correlations_te) > 0.5].drop('receiving_TD').sort_values(ascending=False))

print(breaker)


# Create a single DataFrame with all three correlation series as columns
correlations_df = pd.concat([correlations_all.drop('receiving_TD'), correlations_wr.drop('receiving_TD'), correlations_te.drop('receiving_TD')], axis=1)
correlations_df.columns = ['All Positions', 'WR', 'TE']

# Sort the correlations in descending order (highest to lowest)
correlations_df = correlations_df.sort_values(by='All Positions', ascending=False)

# Set the font size for the annotations
plt.rcParams['font.size'] = 10

# Create heatmaps
plt.figure(figsize=(20, 10))  # Increased width to 20
sns.heatmap(correlations_df, annot=True, cmap='coolwarm', square=False, fmt='.2f', cbar_kws={'shrink': 0.7})
plt.title('Correlations with receiving_TD')
plt.tight_layout()  # Add some space between subplots
plt.show()