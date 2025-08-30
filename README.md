
## Run
要运行文件，在项目根目下运行指令：
```
python -m Zagreus.scripts.train_tracktransferVer2
python -m Zagreus.scripts.train_trackagileVer2
python -m Zagreus.scripts.train_trackagileVer3

python -m Zagreus.scripts.train_dynamics

python -m Zagreus.scripts.test_trackagileVer1
python -m Zagreus.scripts.test_trackagileVer1_onboard
python -m Zagreus.scripts.test_trackagileVer4_onboard
python -m Zagreus.scripts.test_trackagileVer6
python -m Zagreus.scripts.test_trackagileVer6_onboard
```

```
export LD_LIBRARY_PATH=/home/zim/.conda/envs/rlgpu/lib

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH
export PATH=/usr/local/bin:$PATH
make px4_sitl gazebo
```

## Visulization

```
tensorboard --logdir=/home/zim/Documents/Zagreus/Zagreus/test_runs --port 6013
```

## Debug
安装环境后遇到报错：
```
ImportError: /home/core/.conda/envs/rlgpu/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```
请参考链接：https://github.com/pytorch/pytorch/issues/

## Dataset

时间间隔0.02

全部为那啥坐标系

包含信息：
- 世界位置xyz
- 世界速度vx vy vz
- 世界姿态角roll pitch yaw
- 世界角速度droll dpitch dyaw
- 角速度控制指令cdroll cdpitch cdyaw thrust

存储成json，value是np array

```
dataset = {
    "timestamp": time_samples,
    "x": x, "y": y, "z": z,
    "vx": vx, "vy": vy, "vz": vz,
    "roll": roll, "pitch": pitch, "yaw": yaw,
    "droll": droll, "dpitch": dpitch, "dyaw": dyaw,
    "droll_cmd": droll_cmd, "dpitch_cmd": dpitch_cmd, "dyaw_cmd": dyaw_cmd,
    "thrust": thrust
}
```

- 已从北东地转换为东北天
- 已通过numpy插值保证相邻时间步长为0.02

```
python -m Zagreus.dataset.dataset_generaterVer2.py

```
