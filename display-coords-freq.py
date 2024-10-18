import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

def plot_frequency_domain(npz_file):
    # 从 .npz 文件中读取数据
    data = np.load(npz_file)
    coords = data['coords_array']

    # 分离数据
    x_data = coords[:, 0]
    y_data = coords[:, 1]
    z_data = coords[:, 2]
    rx_data = coords[:, 3]
    ry_data = coords[:, 4]
    rz_data = coords[:, 5]

    # 计算采样频率和时间间隔
    sampling_interval = 0.1  # 100ms
    sampling_freq = 1 / sampling_interval
    n_points = len(x_data)

    # 进行傅立叶变换
    freq_x = np.fft.fft(x_data)
    freq_y = np.fft.fft(y_data)
    freq_z = np.fft.fft(z_data)
    freq_rx = np.fft.fft(rx_data)
    freq_ry = np.fft.fft(ry_data)
    freq_rz = np.fft.fft(rz_data)

    # 计算频率轴
    freqs = np.fft.fftfreq(n_points, d=sampling_interval)

    # 取绝对值并只取正频率部分
    freq_x_magnitude = np.abs(freq_x)
    freq_y_magnitude = np.abs(freq_y)
    freq_z_magnitude = np.abs(freq_z)
    freq_rx_magnitude = np.abs(freq_rx)
    freq_ry_magnitude = np.abs(freq_ry)
    freq_rz_magnitude = np.abs(freq_rz)

    # 创建图形
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制位置数据的频率域图
    axs[0].plot(freqs[:n_points // 2], freq_x_magnitude[:n_points // 2], label='X Frequency', color='r')
    axs[0].plot(freqs[:n_points // 2], freq_y_magnitude[:n_points // 2], label='Y Frequency', color='g')
    axs[0].plot(freqs[:n_points // 2], freq_z_magnitude[:n_points // 2], label='Z Frequency', color='b')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].set_title('Position Frequency Domain')
    axs[0].legend()
    axs[0].grid()

    # 绘制旋转数据的频率域图
    axs[1].plot(freqs[:n_points // 2], freq_rx_magnitude[:n_points // 2], label='Rotation X Frequency', color='m')
    axs[1].plot(freqs[:n_points // 2], freq_ry_magnitude[:n_points // 2], label='Rotation Y Frequency', color='c')
    axs[1].plot(freqs[:n_points // 2], freq_rz_magnitude[:n_points // 2], label='Rotation Z Frequency', color='y')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_title('Rotation Frequency Domain')
    axs[1].legend()
    axs[1].grid()

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot frequency domain of coordinates from NPZ file.')
    parser.add_argument('npz_file', help='Path to the NPZ file')
    args = parser.parse_args()

    plot_frequency_domain(args.npz_file)

if __name__ == "__main__":
    main()
