import os
import yaml
import numpy as np


def run_preprocess():
    # Dummy preprocessing: load data from data/dummy_input.txt if exists, otherwise create random data
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    input_file = os.path.join(data_dir, 'dummy_input.txt')
    output_file = os.path.join(data_dir, 'processed_data.txt')
    
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            data = f.read()
        print('[preprocess.py] Loaded data from dummy_input.txt')
    else:
        # Create dummy data
        data = ' '.join([str(x) for x in np.random.randn(100)])
        with open(input_file, 'w') as f:
            f.write(data)
        print('[preprocess.py] Created dummy_input.txt')
    
    # Dummy normalization: convert numbers and scale them between 0 and 1
    numbers = np.array([float(x) for x in data.split()])
    min_val, max_val = numbers.min(), numbers.max()
    norm_numbers = (numbers - min_val) / (max_val - min_val + 1e-8)
    np.savetxt(output_file, norm_numbers, fmt='%.6f')
    print(f'[preprocess.py] Processed data saved to {output_file}')

if __name__ == '__main__':
    run_preprocess()
