from flight_environment import FlightEnvironment
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 假设path_planner.py在当前目录下 
from path_planner import plan_path # 导入路径规划器
from trajectory_generator import TrajectoryGenerator # 导入轨迹生成器模块

# 创建试验环境
env = FlightEnvironment(55)
start = (1,2,0)
goal = (18,18,3)

print("=" * 60)
print("三维飞行环境路径规划演示")
print("=" * 60)
print()

# 验证起点和终点
print(f"起点: {start}")
print(f"终点: {goal}")
print()

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.
# --------------------------------------------------------------------------------------------------- #

print("开始路径规划...")
print()

try:
    # 调用路径规划算法
    path = plan_path(env, start, goal)
    
    if path is not None and len(path) > 0:
        print(f"路径规划完成！找到包含 {len(path)} 个点的路径")
        print(path)
        
        # 计算路径长度
        def calculate_path_length(path_array):
            length = 0.0
            for i in range(len(path_array) - 1):
                dx = path_array[i+1, 0] - path_array[i, 0]
                dy = path_array[i+1, 1] - path_array[i, 1]
                dz = path_array[i+1, 2] - path_array[i, 2]
                length += np.sqrt(dx*dx + dy*dy + dz*dz)
            return length
        
        path_length = calculate_path_length(path)
        print(f"路径总长度: {path_length:.2f} 米")
        print()
        
        # 显示路径信息
        print("路径点坐标:")
        for i, point in enumerate(path):
            print(f"  点 {i:3d}: ({point[0]:6.2f}, {point[1]:6.2f}, {point[2]:6.2f})")
        
        # 检查路径安全性
        print()
        print("检查路径安全性...")
        all_safe = True
        
        # 检查所有点是否在环境内
        for i, point in enumerate(path):
            if env.is_outside(point):
                print(f"警告: 点 {i} {tuple(point)} 超出环境边界")
                all_safe = False
        
        # 检查所有点是否与障碍物碰撞
        for i, point in enumerate(path):
            if env.is_collide(point, epsilon=0.2):
                print(f"警告: 点 {i} {tuple(point)} 与障碍物碰撞")
                all_safe = False
        
        # 检查路径段是否安全
        for i in range(len(path) - 1):
            # 简单检查：在路径段上采样几个点
            p1 = path[i]
            p2 = path[i+1]
            num_samples = max(3, int(np.linalg.norm(p2 - p1) / 0.2))
            
            for j in range(num_samples + 1):
                t = j / num_samples
                sample_point = p1 + t * (p2 - p1)
                if env.is_collide(sample_point, epsilon=0.2):
                    print(f"警告: 路径段 {i}-{i+1} 上的点 {tuple(sample_point)} 与障碍物碰撞")
                    all_safe = False
                    break
        
        if all_safe:
            print("✓ 路径安全检查通过")
        else:
            print("⚠ 路径安全检查未通过，路径可能不安全")
        
        print()
        print("生成可视化...")
        
        # 可视化结果
        env.plot_cylinders(path)
        
    else:
        print("错误: 未找到有效路径")
        
except Exception as e:
    print(f"路径规划过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
    
    # 显示基本的环境信息
    print()
    print("环境信息:")
    print(f"  尺寸: {env.env_width} × {env.env_length} × {env.env_height}")
    print(f"  障碍物数量: {len(env.cylinders)}")
    
    # 仍然尝试可视化环境
    print("显示环境（无路径）...")
    env.plot_cylinders()

print()
print("路径规划程序结束")
print("=" * 60)

# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.
# --------------------------------------------------------------------------------------------------- #
print("开始轨迹生成...")
print("=" * 60)
print()
traj = TrajectoryGenerator(copy.deepcopy(path), 1.0) # 传入已规划完成的路径点（深拷贝），并指定参考速度1m/s
traj.linear_interpolate_path(0.8) # 对于距离大于一定阈值（单位m）的路径点，要进行线性插值以补充中间点

print("插值更新后的路径点坐标:")
for i, point in enumerate(traj.path_points): # 显示路径信息
    print(f"  点 {i:3d}: ({point[0]:6.2f}, {point[1]:6.2f}, {point[2]:6.2f})")

traj.TrajectorySolve() # 轨迹求解 
traj.PlotTrajectory(0.05) # 绘制轨迹随时间变化的曲线图 (0.05s绘制一个点)

traj_dens = traj.traj_dict_dens
all_safe = True
for i in range(len(traj_dens['x'])) : # 需要再次检查所有细化后的轨迹点是否与障碍物碰撞
    point = np.array([traj_dens['x'][i],traj_dens['y'][i],traj_dens['z'][i]])
    if env.is_collide(point, epsilon=0.2):
        print(f"警告: 坐标位于 {tuple(point)} 的点与障碍物碰撞")
        all_safe = False

if all_safe:
    print("✓ 轨迹安全检查通过")
else:
    print("⚠ 轨迹安全检查未通过，生成的轨迹可能不安全")

print()
print("轨迹规划程序结束")
print("=" * 60)

# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
