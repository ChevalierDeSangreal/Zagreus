import gymnasium as gym
from gymnasium import spaces
import numpy as np

class KephaleEnvVer0(gym.Env):
    """示例自定义 Gym 环境模板"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # 渲染模式
        self.render_mode = render_mode

        # 定义动作空间：这里示例为连续动作，区间 [-1, 1] 的 1 维空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 定义观测（状态）空间：这里示例为连续观测，区间 [-inf, inf] 的 3 维空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        # 环境内部状态
        self.state = None
        self.step_count = 0
        self.max_steps = 200

    def reset(self, *, seed=None, options=None):
        # 设置随机种子（可选）
        super().reset(seed=seed)

        # 初始化内部状态
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.step_count = 0

        # 返回初始观测、info 字典
        observation = self.state.copy()
        info = {}
        return observation, info

    def step(self, action):
        # 应用动作到环境动力学，更新状态
        # —— 这里仅作示例：将状态增加动作值
        self.state = self.state + action

        # 计算奖励：示例为负的状态范数
        reward = -np.linalg.norm(self.state)

        # 判断是否结束
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # 渲染信息（可选）
        info = {}

        # 返回：观测、奖励、是否结束、截断、info
        return self.state.copy(), reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            # 返回一帧 RGB 图像（numpy 数组）
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
            # … 在此绘制 frame …
            return frame
        elif self.render_mode == "human":
            # 可视化窗口或打印
            print(f"Step {self.step_count}, State: {self.state}")

    def close(self):
        # 清理资源（如打开的窗口等）
        pass

# 使用示例
if __name__ == "__main__":
    env = KephaleEnvVer0(render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        env.render()
    env.close()
