import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def generate_theoretical_line(D, N):
    """生成理论直线上的点
    D: 直线长度
    N: 点的数量
    """
    t = np.linspace(0, 1, N)
    x = D * t
    y = np.zeros_like(x)
    return np.column_stack((x, y))

def generate_actual_line(theoretical_points, mean_dev, std_dev):
    """生成带有随机偏差的实际直线"""
    noise = np.random.normal(0, std_dev, theoretical_points.shape)
    bias = np.tile(mean_dev, (len(theoretical_points), 1))
    return theoretical_points + noise + bias

def plot_comparison(theoretical_points, actual_points, point_diameter, std_dev):
    """绘制理论直线和实际直线的对比图"""
    fig = plt.figure(figsize=(12, 12))
    
    # 2D视图（上图）
    ax1 = fig.add_subplot(211)
    
    # 绘制理论点和实际点
    for i in range(len(theoretical_points)):
        circle_theo = plt.Circle(theoretical_points[i], point_diameter/2, 
                               color='blue', alpha=0.3, label='Theoretical' if i == 0 else "")
        ax1.add_patch(circle_theo)
        
        circle_actual = plt.Circle(actual_points[i], point_diameter/2, 
                                 color='red', alpha=0.3, label='Actual' if i == 0 else "")
        ax1.add_patch(circle_actual)
    
    ax1.plot(theoretical_points[:, 0], theoretical_points[:, 1], 'b--', alpha=0.5)
    ax1.plot(actual_points[:, 0], actual_points[:, 1], 'r--', alpha=0.5)
    
    ax1.set_aspect('equal')
    
    margin = point_diameter
    ax1.set_xlim(min(theoretical_points[:, 0].min(), actual_points[:, 0].min()) - margin,
                 max(theoretical_points[:, 0].max(), actual_points[:, 0].max()) + margin)
    ax1.set_ylim(min(theoretical_points[:, 1].min(), actual_points[:, 1].min()) - margin * 10,
                 max(theoretical_points[:, 1].max(), actual_points[:, 1].max()) + margin * 10)
    
    ax1.set_title(f'2D View (Point Diameter: {point_diameter}mm, Std Dev: {std_dev}mm)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    
    # 计算误差统计
    errors = np.linalg.norm(actual_points - theoretical_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)
    mean_deviation = np.mean(actual_points - theoretical_points, axis=0)
    
    # 误差直方图（下图）
    ax2 = fig.add_subplot(212)
    ax2.hist(errors, bins=20, alpha=0.7)
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Error Magnitude')
    ax2.set_ylabel('Count')
    
    stats_text = (f'Mean Error: {mean_error:.3f}\n'
                 f'Max Error: {max_error:.3f}\n'
                 f'Std Dev: {std_error:.3f}\n'
                 f'Mean X Deviation: {mean_deviation[0]:.3f}\n'
                 f'Mean Y Deviation: {mean_deviation[1]:.3f}')
    ax2.text(0.95, 0.95, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 文件名包含参数信息
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'line_comparison_d{point_diameter}_std{std_dev}_{timestamp}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    
    return mean_error, max_error, std_error, mean_deviation

def main():
    parser = argparse.ArgumentParser(description='Compare theoretical and actual lines with given parameters.')
    parser.add_argument('-d', '--diameter', type=float, default=1.0,
                       help='Point diameter in mm (default: 1.0)')
    parser.add_argument('-s', '--std_dev', type=float, default=0.1,
                       help='Standard deviation of the error in mm (default: 0.1)')
    parser.add_argument('--length', type=float, default=100.0,
                       help='Line length in mm (default: 100.0)')
    parser.add_argument('-n', '--num_points', type=int, default=100,
                       help='Number of points (default: 100)')
    args = parser.parse_args()
    
    # 参数设置
    D = args.length
    N = args.num_points
    mean_dev = [0.0, 0.0]  # 偏差的均值 [x方向, y方向] (mm)
    std_dev = args.std_dev
    point_diameter = args.diameter
    
    # 生成理论直线
    theoretical_points = generate_theoretical_line(D, N)
    
    # 生成实际直线
    actual_points = generate_actual_line(theoretical_points, mean_dev, std_dev)
    
    # 绘制对比图并计算统计信息
    mean_error, max_error, std_error, mean_deviation = plot_comparison(
        theoretical_points, actual_points, point_diameter, std_dev)
    
    # 打印统计信息
    print(f"Line Comparison Statistics:")
    print(f"Line Length: {D} mm")
    print(f"Number of Points: {N}")
    print(f"Point Diameter: {point_diameter} mm")
    print(f"Standard Deviation: {std_dev} mm")
    print(f"Mean Deviation X: {mean_deviation[0]:.3f} mm")
    print(f"Mean Deviation Y: {mean_deviation[1]:.3f} mm")
    print(f"Mean Error: {mean_error:.3f} mm")
    print(f"Max Error: {max_error:.3f} mm")
    print(f"Standard Deviation: {std_error:.3f} mm")
    
    # 保存数据
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    filename = f'line_data_d{point_diameter}_std{std_dev}_{timestamp}.npz'
    np.savez(filename,
             theoretical_points=theoretical_points,
             actual_points=actual_points,
             point_diameter=point_diameter,
             mean_deviation=mean_deviation,
             errors=np.linalg.norm(actual_points - theoretical_points, axis=1))

if __name__ == "__main__":
    main()
