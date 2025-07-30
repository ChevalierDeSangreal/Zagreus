
## Run
要运行文件，在项目根目录上级目录下运行指令：
```
python -m Kephale.scripts.train_tracktransferVer2
python -m Kephale.scripts.test_trackagileVer1
```
## Debug
安装环境后遇到报错：
```
ImportError: /home/core/.conda/envs/rlgpu/lib/python3.7/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```
请参考链接：https://github.com/pytorch/pytorch/issues/123097