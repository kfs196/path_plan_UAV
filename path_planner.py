"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""

import numpy as np
import math
import random
import heapq
from typing import List, Tuple, Dict, Optional

class AdaptiveProbabilityTree:
    
    
    def __init__(self, env, safety_margin: float = 0.3):
        """
        初始化路径规划器
        
        参数:
            env: FlightEnvironment对象
            safety_margin: 安全裕度（米）
        """
        self.env = env
        self.safety_margin = safety_margin
        
        # 算法参数
        self.max_iterations = 8000
        self.step_size = 1.2  # 基础步长
        self.goal_bias = 0.35  # 初始目标偏置概率
        self.random_bias = 0.4  # 随机探索概率
        self.retreat_bias = 0.25  # 回退探索概率
        
        # 自适应参数调整
        self.adaptive_factor = 1.0
        self.successive_failures = 0
        
    def euclidean_distance(self, p1: Tuple[float, float, float], 
                          p2: Tuple[float, float, float]) -> float:
        """计算欧几里得距离"""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def is_point_safe(self, point: Tuple[float, float, float]) -> bool:
        """检查点是否安全"""
        # 检查边界
        if self.env.is_outside(point):
            return False
        
        # 检查碰撞，增加安全裕度
        if self.env.is_collide(point, epsilon=self.safety_margin):
            return False
        
        return True
    
    def sample_point(self, current: Tuple[float, float, float], 
                    goal: Tuple[float, float, float]) -> Tuple[float, float, float]:
       
        # 自适应调整采样概率
        rand_val = random.random()
        
        # 如果连续失败次数多，增加随机探索
        adaptive_goal_bias = self.goal_bias * (1.0 - min(self.successive_failures * 0.1, 0.5))
        adaptive_random_bias = self.random_bias * (1.0 + min(self.successive_failures * 0.1, 0.5))
        
        # 策略1: 目标导向采样
        if rand_val < adaptive_goal_bias:
            # 向目标方向采样，但加入随机扰动
            direction = (
                goal[0] - current[0],
                goal[1] - current[1],
                goal[2] - current[2]
            )
            dist = max(self.euclidean_distance(current, goal), 0.001)
            
            # 归一化并添加扰动
            perturbation = random.uniform(0.8, 1.2)
            sample = (
                current[0] + direction[0]/dist * self.step_size * perturbation * 2,
                current[1] + direction[1]/dist * self.step_size * perturbation * 2,
                current[2] + direction[2]/dist * self.step_size * perturbation * 2
            )
            
        # 策略2: 随机探索采样
        elif rand_val < adaptive_goal_bias + adaptive_random_bias:
            # 在整个空间随机采样
            sample = (
                random.uniform(0, self.env.env_width),
                random.uniform(0, self.env.env_length),
                random.uniform(0, self.env.env_height)
            )
            
        # 策略3: 回退采样（在当前位置周围小范围探索）
        else:
            # 在失败区域周围探索
            radius = self.step_size * (1.0 + self.successive_failures * 0.2)
            angle = random.uniform(0, 2 * math.pi)
            height_variation = random.uniform(-self.step_size * 0.5, self.step_size * 0.5)
            
            sample = (
                current[0] + math.cos(angle) * radius,
                current[1] + math.sin(angle) * radius,
                current[2] + height_variation
            )
        
        # 确保采样点在边界内
        sample = (
            max(0, min(sample[0], self.env.env_width)),
            max(0, min(sample[1], self.env.env_length)),
            max(0, min(sample[2], self.env.env_height))
        )
        
        return sample
    
    def interpolate_segment(self, start: Tuple[float, float, float],
                           end: Tuple[float, float, float],
                           resolution: float = 0.2) -> List[Tuple[float, float, float]]:
        """在两点之间生成插值点用于碰撞检测"""
        distance = self.euclidean_distance(start, end)
        if distance == 0:
            return []
        
        num_points = max(2, int(distance / resolution))
        points = []
        
        for i in range(num_points + 1):
            t = i / num_points
            point = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2])
            )
            points.append(point)
        
        return points
    
    def is_segment_safe(self, start: Tuple[float, float, float],
                       end: Tuple[float, float, float]) -> bool:
        """检查路径段是否安全"""
        # 快速检查端点
        if not self.is_point_safe(start) or not self.is_point_safe(end):
            return False
        
        # 沿线段采样检查
        points = self.interpolate_segment(start, end, resolution=0.15)
        for point in points:
            if self.env.is_collide(point, epsilon=self.safety_margin):
                return False
        
        return True
    
    def find_nearest_node(self, tree_nodes: List[Tuple[float, float, float]], 
                         sample_point: Tuple[float, float, float]) -> int:
        """在树中找到距离采样点最近的节点"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(tree_nodes):
            dist = self.euclidean_distance(node, sample_point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def steer_towards(self, from_point: Tuple[float, float, float],
                     to_point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """从起点向目标点方向移动一步"""
        direction = (
            to_point[0] - from_point[0],
            to_point[1] - from_point[1],
            to_point[2] - from_point[2]
        )
        
        distance = max(self.euclidean_distance(from_point, to_point), 0.001)
        
        # 自适应步长：根据连续失败次数调整
        adaptive_step = self.step_size * (1.0 - min(self.successive_failures * 0.1, 0.3))
        
        # 如果距离小于步长，直接返回目标点
        if distance <= adaptive_step:
            return to_point
        
        # 否则按比例移动
        step_factor = adaptive_step / distance
        new_point = (
            from_point[0] + direction[0] * step_factor,
            from_point[1] + direction[1] * step_factor,
            from_point[2] + direction[2] * step_factor
        )
        
        return new_point
    
    def optimize_path_segment(self, start: Tuple[float, float, float],
                            end: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """
        优化路径段：尝试在两点之间找到更短的安全路径
        
        方法：尝试不同的中间高度，找到最短的安全路径
        """
        # 如果直接连接安全，直接返回
        if self.is_segment_safe(start, end):
            return [start, end]
        
        # 尝试不同的高度策略
        strategies = [
            # 策略1: 保持平均高度
            lambda t: (start[2] + end[2]) / 2,
            # 策略2: 逐渐上升然后下降
            lambda t: start[2] + (end[2] - start[2]) * t + math.sin(t * math.pi) * 1.0,
            # 策略3: 尝试较高高度
            lambda t: max(start[2], end[2]) + 1.0,
            # 策略4: 尝试较低高度
            lambda t: min(start[2], end[2]) - 0.5,
        ]
        
        best_path = None
        min_length = float('inf')
        
        for strategy in strategies:
            # 创建中间点
            mid_point = (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
                strategy(0.5)
            )
            
            # 确保中间点安全
            if not self.is_point_safe(mid_point):
                continue
            
            # 检查两段路径是否安全
            if (self.is_segment_safe(start, mid_point) and 
                self.is_segment_safe(mid_point, end)):
                
                path_length = (self.euclidean_distance(start, mid_point) + 
                              self.euclidean_distance(mid_point, end))
                
                if path_length < min_length:
                    min_length = path_length
                    best_path = [start, mid_point, end]
        
        return best_path
    
    def smooth_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """平滑路径：去除冗余点，优化路径质量"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 尝试跳过尽可能多的点
            best_j = i + 1
            for j in range(i + 2, len(path)):
                if self.is_segment_safe(path[i], path[j]):
                    best_j = j
                else:
                    break
            
            if best_j > i + 1:
                # 尝试优化这一段
                optimized_segment = self.optimize_path_segment(path[i], path[best_j])
                if optimized_segment:
                    # 去掉第一个点（已经在smoothed中）
                    smoothed.extend(optimized_segment[1:])
                else:
                    smoothed.append(path[best_j])
                i = best_j
            else:
                smoothed.append(path[i + 1])
                i += 1
        
        return smoothed
    
    def find_path(self, start: Tuple[float, float, float], 
                 goal: Tuple[float, float, float]) -> Optional[np.ndarray]:
        """
        主路径规划函数
        
        返回:
            N×3的numpy数组，表示路径点坐标，或None（如果找不到路径）
        """
        # 验证起点和终点
        if not self.is_point_safe(start):
            print(f"错误：起点 {start} 不安全")
            return None
        
        if not self.is_point_safe(goal):
            print(f"错误：终点 {goal} 不安全")
            return None
        
        print(f"开始路径规划: {start} -> {goal}")
        print(f"环境尺寸: {self.env.env_width}×{self.env.env_length}×{self.env.env_height}")
        print(f"障碍物数量: {len(self.env.cylinders)}")
        
        # 初始化树结构
        tree_nodes = [start]  # 节点列表
        tree_parents = [-1]   # 父节点索引列表（-1表示根节点）
        
        # 检查是否可以直接连接
        if self.is_segment_safe(start, goal):
            print("可以直接连接起点和终点")
            path = [start, goal]
            return np.array(path)
        
        # 主搜索循环
        for iteration in range(self.max_iterations):
            if iteration % 1000 == 0 and iteration > 0:
                print(f"已迭代 {iteration} 次，树中有 {len(tree_nodes)} 个节点")
            
            # 自适应采样
            current_node = tree_nodes[-1]  # 使用最近添加的节点作为当前节点
            sample_point = self.sample_point(current_node, goal)
            
            # 找到最近的节点
            nearest_idx = self.find_nearest_node(tree_nodes, sample_point)
            nearest_node = tree_nodes[nearest_idx]
            
            # 向采样点方向扩展
            new_node = self.steer_towards(nearest_node, sample_point)
            
            # 检查新节点是否安全且路径段安全
            if (self.is_point_safe(new_node) and 
                self.is_segment_safe(nearest_node, new_node)):
                
                # 添加到树中
                tree_nodes.append(new_node)
                tree_parents.append(nearest_idx)
                
                # 重置连续失败计数
                self.successive_failures = max(0, self.successive_failures - 1)
                
                # 检查是否能连接到目标
                if self.is_segment_safe(new_node, goal):
                    print(f"在第 {iteration} 次迭代中找到路径")
                    
                    # 重建路径
                    path_nodes = [goal]
                    node_idx = len(tree_nodes) - 1
                    
                    while node_idx != -1:
                        path_nodes.append(tree_nodes[node_idx])
                        node_idx = tree_parents[node_idx]
                    
                    path_nodes.reverse()
                    
                    # 平滑路径
                    smoothed_path = self.smooth_path(path_nodes)
                    
                    # 确保路径是安全的
                    if self.validate_path(smoothed_path):
                        print(f"原始路径点: {len(path_nodes)}，平滑后: {len(smoothed_path)}")
                        print(f"路径长度: {self.calculate_path_length(smoothed_path):.2f} 米")
                        return np.array(smoothed_path)
                    else:
                        print("警告：平滑后的路径验证失败")
                
            else:
                # 扩展失败，增加连续失败计数
                self.successive_failures += 1
            
            # 如果连续失败次数过多，随机重启
            if self.successive_failures > 50:
                # 从随机节点重新开始
                if len(tree_nodes) > 10:
                    random_node = random.choice(tree_nodes[-10:])
                    tree_nodes.append(random_node)
                    tree_parents.append(len(tree_nodes) - 2)
                    self.successive_failures = 0
        
        print(f"在 {self.max_iterations} 次迭代后未找到路径")
        
        # 尝试最后的简化方法
        return self.fallback_path(start, goal)
    
    def fallback_path(self, start: Tuple[float, float, float],
                     goal: Tuple[float, float, float]) -> Optional[np.ndarray]:
        """备用路径规划方法：当主算法失败时使用"""
        print("使用备用路径规划方法...")
        
        # 尝试简单的中间点策略
        mid_points = [
            # 中间点1：空间中心
            (self.env.env_width / 2, self.env.env_length / 2, self.env.env_height / 2),
            # 中间点2：较高点
            (self.env.env_width / 2, self.env.env_length / 2, self.env.env_height * 0.8),
            # 中间点3：起点和终点的平均值
            ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2, (start[2] + goal[2]) / 2),
        ]
        
        for mid_point in mid_points:
            if (self.is_point_safe(mid_point) and
                self.is_segment_safe(start, mid_point) and
                self.is_segment_safe(mid_point, goal)):
                
                path = [start, mid_point, goal]
                print("备用方法找到路径")
                return np.array(path)
        
        # 尝试简单的爬行方法
        print("尝试爬行方法...")
        path = self.crawl_path(start, goal)
        if path and self.validate_path(path):
            return np.array(path)
        
        return None
    
    def crawl_path(self, start: Tuple[float, float, float],
                  goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """爬行路径方法：沿着直线小心前进，遇到障碍物时绕行"""
        path = [start]
        current = start
        max_steps = 200
        step = 0.5  # 小步长
        
        for _ in range(max_steps):
            # 计算到目标的方向
            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            dz = goal[2] - current[2]
            dist = max(self.euclidean_distance(current, goal), 0.001)
            
            # 如果很近，尝试直接连接
            if dist < step * 2:
                if self.is_segment_safe(current, goal):
                    path.append(goal)
                    break
            
            # 尝试向目标方向移动
            direction = (dx/dist, dy/dist, dz/dist)
            next_point = (
                current[0] + direction[0] * step,
                current[1] + direction[1] * step,
                current[2] + direction[2] * step
            )
            
            # 如果下一个点安全，则移动
            if (self.is_point_safe(next_point) and 
                self.is_segment_safe(current, next_point)):
                
                path.append(next_point)
                current = next_point
            else:
                # 尝试绕行：向上移动
                up_point = (current[0], current[1], current[2] + step)
                if (self.is_point_safe(up_point) and 
                    self.is_segment_safe(current, up_point)):
                    
                    path.append(up_point)
                    current = up_point
                else:
                    # 尝试随机方向
                    for _ in range(10):
                        random_dir = (
                            random.uniform(-1, 1),
                            random.uniform(-1, 1),
                            random.uniform(0, 0.5)  # 倾向于向上
                        )
                        dir_len = math.sqrt(sum(d*d for d in random_dir))
                        if dir_len > 0:
                            random_dir = (d/dir_len for d in random_dir)
                            random_point = (
                                current[0] + random_dir[0] * step,
                                current[1] + random_dir[1] * step,
                                current[2] + random_dir[2] * step
                            )
                            
                            if (self.is_point_safe(random_point) and 
                                self.is_segment_safe(current, random_point)):
                                
                                path.append(random_point)
                                current = random_point
                                break
        
        # 最后尝试连接到目标
        if self.is_segment_safe(current, goal):
            path.append(goal)
        
        return path
    
    def calculate_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """计算路径长度"""
        length = 0.0
        for i in range(len(path) - 1):
            length += self.euclidean_distance(path[i], path[i + 1])
        return length
    
    def validate_path(self, path: List[Tuple[float, float, float]]) -> bool:
        """验证整个路径是否安全"""
        if not path:
            return False
        
        # 检查所有点是否安全
        for point in path:
            if not self.is_point_safe(point):
                print(f"路径点 {point} 不安全")
                return False
        
        # 检查所有路径段是否安全
        for i in range(len(path) - 1):
            if not self.is_segment_safe(path[i], path[i + 1]):
                print(f"路径段 {i} 到 {i+1} 不安全")
                return False
        
        return True


# 主路径规划函数（提供给main.py调用）
def plan_path(env, start: Tuple[float, float, float], 
              goal: Tuple[float, float, float]) -> np.ndarray:
    """
    路径规划主接口函数
    
    Parameters:
        env: FlightEnvironment对象
        start: 起点坐标 (x, y, z)
        goal: 终点坐标 (x, y, z)
    
    Returns:
        N×3的numpy数组，表示路径点坐标
    """
    # 创建路径规划器
    planner = AdaptiveProbabilityTree(env, safety_margin=0.3)
    
    # 规划路径
    path = planner.find_path(start, goal)
    
    if path is None:
        print("路径规划失败")
 
        return np.array([start, goal])
    
    return path











