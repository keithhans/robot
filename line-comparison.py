import numpy as np
import matplotlib.pyplot as plt
import time

def generate_theoretical_line(D, N):
    """生成理论直线上的点
    D: 直线长度
    N: 点的数量
    """
    # 生成均匀分布的点
    t = np.linspace(0, 1, N)
    x = D * t
    y = np.zeros_like(x)
    
    return np.column_stack((x, y))

def generate_actual_line(theoretical_points, mean_dev, std_dev):
    """生成带有随机偏差的实际直线
    theoretical_points: 理论点的坐标
    mean_dev: 偏差的均值 [mean_x, mean_y]
    std_dev: 偏差的标准差
    """
    # 为每个坐标添加正态分布的随机偏差
    noise = np.random.normal(0, std_dev, theoretical_points.shape)
    # 添加均值偏差
    bias = np.tile(mean_dev, (len(theoretical_points), 1))
    return theoretical_points + noise + bias

def plot_comparison(theoretical_points, actual_points, point_diameter):
    """绘制理论直线和实际直线的对比图"""
    # 创建上下布局的图形，增大尺寸
    fig = plt.figure(figsize=(12, 12))
    
    # 2D视图（上图）
    ax1 = fig.add_subplot(211)
    
    # 绘制理论点和实际点（用圆表示）
    for i in range(len(theoretical_points)):
        # 理论点（蓝色）
        circle_theo = plt.Circle(theoretical_points[i], point_diameter/2, 
                               color='blue', alpha=0.3, label='Theoretical' if i == 0 else "")
        ax1.add_patch(circle_theo)
        
        # 实际点（红色）
        circle_actual = plt.Circle(actual_points[i], point_diameter/2, 
                                 color='red', alpha=0.3, label='Actual' if i == 0 else "")
        ax1.add_patch(circle_actual)
    
    # 绘制连接线
    ax1.plot(theoretical_points[:, 0], theoretical_points[:, 1], 'b--', alpha=0.5)
    ax1.plot(actual_points[:, 0], actual_points[:, 1], 'r--', alpha=0.5)
    
    # 设置坐标轴等比例
    ax1.set_aspect('equal')
    
    # 设置坐标轴范围（留出点的直径的空间）
    margin = point_diameter
    ax1.set_xlim(min(theoretical_points[:, 0].min(), actual_points[:, 0].min()) - margin,
                 max(theoretical_points[:, 0].max(), actual_points[:, 0].max()) + margin)
    ax1.set_ylim(min(theoretical_points[:, 1].min(), actual_points[:, 1].min()) - margin * 10,
                 max(theoretical_points[:, 1].max(), actual_points[:, 1].max()) + margin * 10)
    
    ax1.set_title('2D View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    
    # 计算误差统计
    errors = np.linalg.norm(actual_points - theoretical_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)
    
    # 计算X和Y方向的平均偏差
    mean_deviation = np.mean(actual_points - theoretical_points, axis=0)
    
    # 误差直方图（下图）
    ax2 = fig.add_subplot(212)
    ax2.hist(errors, bins=20, alpha=0.7)
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Error Magnitude')
    ax2.set_ylabel('Count')
    
    # 添加误差统计信息
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
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图像
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(f'line_comparison_{timestamp}.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    return mean_error, max_error, std_error, mean_deviation

def main():
    # 参数设置
    D = 100.0  # 直线长度（mm）
    N = 100     # 点的数量
    mean_dev = [0.05, 0.05]  # 偏差的均值 [x方向, y方向] (mm)
    std_dev = 0.05  # 偏差的标准差（mm）
    point_diameter = 1.0  # 点的直径（mm）
    
    # 生成理论直线
    theoretical_points = generate_theoretical_line(D, N)
    
    # 生成实际直线
    actual_points = generate_actual_line(theoretical_points, mean_dev, std_dev)
    
    # 绘制对比图并计算统计信息
    mean_error, max_error, std_error, mean_deviation = plot_comparison(theoretical_points, actual_points, point_diameter)
    
    # 打印统计信息
    print(f"Line Comparison Statistics:")
    print(f"Line Length: {D} mm")
    print(f"Number of Points: {N}")
    print(f"Point Diameter: {point_diameter} mm")
    print(f"Mean Deviation X: {mean_deviation[0]:.3f} mm")
    print(f"Mean Deviation Y: {mean_deviation[1]:.3f} mm")
    print(f"Mean Error: {mean_error:.3f} mm")
    print(f"Max Error: {max_error:.3f} mm")
    print(f"Standard Deviation: {std_error:.3f} mm")
    
    # 保存数据
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    np.savez(f'line_data_{timestamp}.npz',
             theoretical_points=theoretical_points,
             actual_points=actual_points,
             point_diameter=point_diameter,
             mean_deviation=mean_deviation,
             errors=np.linalg.norm(actual_points - theoretical_points, axis=1))

if __name__ == "__main__":
    main()
