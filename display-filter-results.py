import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butter_lowpass_lfilter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_coordinates(coords_array, filtered_coords_array, max_points=200, extra_filter_freq=None):
    num_points = min(len(coords_array), max_points)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    title = f'Comparison of Original and Filtered Coordinates (First {num_points} points)'
    if extra_filter_freq:
        title += f' with {extra_filter_freq}Hz filter'
    fig.suptitle(title, fontsize=16)
    
    labels = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    
    # 假设采样频率为10Hz
    sampling_freq = 10.0
    
    for i, label in enumerate(labels):
        row = i // 2
        col = i % 2
        
        original_data = coords_array[:num_points, i]
        filtered_data = filtered_coords_array[:num_points, i]
        
        axs[row, col].plot(original_data, label='Original', alpha=0.7)
        axs[row, col].plot(filtered_data, label='Filtered (2Hz)', alpha=0.7)
        
        if extra_filter_freq:
            extra_filtered = apply_butter_lowpass_lfilter(original_data, cutoff=extra_filter_freq, fs=sampling_freq)
            axs[row, col].plot(extra_filtered, label=f'Filtered ({extra_filter_freq}Hz)', alpha=0.7)
        
        axs[row, col].set_title(f'{label.upper()} Coordinate')
        axs[row, col].set_xlabel('Sample')
        axs[row, col].set_ylabel('Value')
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Display filtered coordinates from NPZ file.')
    parser.add_argument('npz_file', help='Path to the NPZ file')
    parser.add_argument('-f', '--filter', type=float, help='Additional filter frequency (Hz)')
    parser.add_argument('-m', '--max_points', type=int, default=200, help='Maximum number of points to display')
    args = parser.parse_args()

    try:
        data = np.load(args.npz_file)
        coords_array = data['coords_array']
        filtered_coords_array = data['filtered_coords_array']
    except Exception as e:
        print(f"Error loading the NPZ file: {e}")
        sys.exit(1)

    if coords_array.shape != filtered_coords_array.shape:
        print("Error: The shapes of original and filtered coordinate arrays do not match.")
        sys.exit(1)

    plot_coordinates(coords_array, filtered_coords_array, max_points=args.max_points, extra_filter_freq=args.filter)

if __name__ == "__main__":
    main()
