import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from scipy import signal

def butter_lowpass_scipy(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_python(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    w0 = normal_cutoff * np.pi
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / (2.0 * 0.707)  # Q factor of 0.707 for Butterworth

    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])

    return b, a

def apply_butter_lowpass_filter_python(data, b, a):
    y = np.zeros_like(data)
    x_prev = np.zeros(2)
    y_prev = np.zeros(2)

    for i, x in enumerate(data):
        y[i] = (b[0] * x + b[1] * x_prev[0] + b[2] * x_prev[1]
                - a[1] * y_prev[0] - a[2] * y_prev[1])
        
        x_prev[1] = x_prev[0]
        x_prev[0] = x
        y_prev[1] = y_prev[0]
        y_prev[0] = y[i]

    return y[-1]

def online_filter(data, cutoff, fs, buffer_size=20, use_scipy=False):
    filtered_data = np.zeros_like(data)
    buffer = []
    b, a = butter_lowpass_scipy(cutoff, fs) if use_scipy else butter_lowpass_python(cutoff, fs)
    
    for i, value in enumerate(data):
        buffer.append(value)
        if len(buffer) > buffer_size:
            buffer.pop(0)
        
        if len(buffer) == buffer_size:
            if use_scipy:
                filtered_value = signal.lfilter(b, a, buffer)[-1]
            else:
                filtered_value = apply_butter_lowpass_filter_python(buffer, b, a)
        else:
            filtered_value = value
        
        filtered_data[i] = filtered_value
    
    return filtered_data

def plot_coordinates(coords_array, filtered_coords_array, max_points=200, extra_filter_freq=None, use_scipy=False):
    num_points = min(len(coords_array), max_points)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    title = f'Comparison of Original and Filtered Coordinates (First {num_points} points)'
    if extra_filter_freq:
        title += f' with {extra_filter_freq}Hz filter'
    if use_scipy:
        title += ' (using SciPy lfilter)'
    else:
        title += ' (using Python implementation)'
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
        axs[row, col].plot(filtered_data, label='Filtered', alpha=0.7)
        
        if extra_filter_freq:
            extra_filtered = online_filter(original_data, cutoff=extra_filter_freq, fs=sampling_freq, use_scipy=use_scipy)
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
    parser.add_argument('-s', '--scipy', action='store_true', help='Use SciPy for both filter design and application')
    args = parser.parse_args()

    try:
        data = np.load(args.npz_file)
        coords_array = data['coords_array']
        filtered_coords_array = data['filtered_coords_array']
    except Exception as e:
        print(f"Error loading the NPZ file: {e}")
        sys.exit(1)

    if coords_array.shape != filtered_coords_array.shape:
        print("Error: The shapes of original and filtered coordinate arrays do not match.", coords_array.shape, filtered_coords_array.shape)
        sys.exit(1)

    plot_coordinates(coords_array, filtered_coords_array, max_points=args.max_points, extra_filter_freq=args.filter, use_scipy=args.scipy)

if __name__ == "__main__":
    main()
