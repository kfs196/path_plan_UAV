## 智能移动机器人技术-路径与轨迹规划大作业

* 极简版文档（对应V0.4）

#### 1. 环境配置

本项目只需要在虚拟环境下安装Numpy、Scipy、Matplotlib，共3个第三方库（以下代码为清华源安装）：

```
pip install numpy scipy matplotlib -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

随后即满足正常运行的条件。

#### 2. 程序运行

直接执行main.py即可，随后会依次弹出两个Matplotlib窗体，分别为路径规划与轨迹规划图（注意关闭第一个弹窗后，轨迹规划相关程序才会继续执行；关闭第二个弹窗后，程序才能执行至完毕）。

* 注意：若轨迹规划阶段，出现了“轨迹安全检查未通过”问题，可以通过适当调小main.py中如下代码的参数即可（约150行）：

  ```
  traj.linear_interpolate_path(0.8) # 可改为0.7等
  ```
