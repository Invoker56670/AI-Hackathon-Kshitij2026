import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Column names based on C-MAPSS documentation
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

# Sensors to drop (constant/quasi-constant) as per tips.pdf
# Sensors to KEEP: 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
# Sensors to DROP: 1, 5, 6, 10, 16, 18, 19
DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
DROP_COLS = ['op1', 'op2', 'op3'] + DROP_SENSORS

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLS)
    return train_df, test_df

def calculate_rul_piecewise(df, early_rul=125):
    """
    Calculates RUL for training data.
    Caps RUL at 'early_rul' (piecewise linear degradation).
    """
    # Calculate max cycle for each unit
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Apply piecewise limiting
    df['RUL'] = df['RUL'].clip(upper=early_rul)
    
    return df.drop(columns=['max_cycle'])

def process_data(train_df, test_df, window_size=30):
    """
    Standardize and window the data.
    """
    # Feature Selection
    features = [c for c in train_df.columns if c not in DROP_COLS + ['unit', 'cycle', 'RUL']]
    
    # Normalization (StandardScaler)
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    return train_df, test_df, features, scaler

def gen_sequence(id_df, seq_length, seq_cols):
    """
    Generates sequences for a single unit.
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    
    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    """
    Generates labels for a single unit.
    """
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    
    return data_matrix[seq_length-1:num_elements, :]

def create_sequences(df, features, window_size=30, target_col='RUL'):
    """
    Create sequences for LSTM/Time-series models.
    """
    seq_gen = (list(gen_sequence(df[df['unit'] == id], window_size, features))
               for id in df['unit'].unique())
    
    # Generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    if target_col in df.columns:
        label_gen = (gen_labels(df[df['unit'] == id], window_size, [target_col])
                     for id in df['unit'].unique())
        label_array = np.concatenate(list(label_gen)).astype(np.float32)
        return seq_array, label_array
    
    return seq_array

if __name__ == "__main__":
    TRAIN_FILE = 'train_FD001.txt/train_FD001.txt'
    TEST_FILE = 'test_FD001.txt/test_FD001.txt'
    
    print("Loading data...")
    train, test = load_data(TRAIN_FILE, TEST_FILE)
    
    print("Calculating RUL...")
    train = calculate_rul_piecewise(train)
    
    print("Processing data...")
    train, test, feats, scaler = process_data(train, test)
    
    print(f"Features: {feats}")
    
    print("Creating sequences...")
    X_train, y_train = create_sequences(train, feats)
    print(f"Train Shape: {X_train.shape}, {y_train.shape}")
    
    # For test set, we usually take the LAST sequence for each engine to predict final RUL
    # But for full evaluation we might need more.
    # The challenge says "run your model on the test_FD001.txt".
    # Usually test_FD001.txt is cut off at some point prior to failure.
    # We will generate sequences for the test set as well.
    # IMPORTANT: We only care about the *last* sequence for prediction in typical C-MAPSS usage, 
    # but let's see what the challenge implies. 
    # "Run your model on the test... result". 
    # Usually this implies predicting RUL for the *last* recorded cycle of each unit.
    
    seq_gen_test = [list(gen_sequence(test[test['unit'] == id], 30, feats)) for id in test['unit'].unique()]
    # Note: Some units might be shorter than 30 cycles? tips.pdf says min test cycle is 31.
    
    # We will just grab the last sequence for each unit for the final prediction
    X_test_last = []
    ids = []
    for unit_id in test['unit'].unique():
        unit_data = test[test['unit'] == unit_id]
        if len(unit_data) >= 30:
            # Take the last window
            seq = unit_data[feats].values[-30:]
            X_test_last.append(seq)
            ids.append(unit_id)
            
    X_test_last = np.array(X_test_last).astype(np.float32)
    print(f"Test Shape (Last window per unit): {X_test_last.shape}")
