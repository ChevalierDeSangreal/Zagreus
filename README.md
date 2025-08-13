
## Run
要运行文件，在项目根目下运行指令：
```
python -m Zagreus.scripts.train_tracktransferVer2
python -m Zagreus.scripts.train_trackagileVer2
python -m Zagreus.scripts.train_trackagileVer3

python -m Zagreus.scripts.test_trackagileVer1
python -m Zagreus.scripts.test_trackagileVer1_onboard
python -m Zagreus.scripts.test_trackagileVer4_onboard
python -m Zagreus.scripts.test_trackagileVer6
python -m Zagreus.scripts.test_trackagileVer6_onboard
```

```
export LD_LIBRARY_PATH=/home/zim/.conda/envs/rlgpu/lib

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

