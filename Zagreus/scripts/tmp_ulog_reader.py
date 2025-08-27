from pyulog import ULog


"""

"""

# 读取文件
ulog_file = '/home/core/wangzimo/Zagreus/Zagreus/data/06_46_55.ulg'
ulog = ULog(ulog_file)

# for dataset in ulog.data_list:
#     print(dataset.name)

# vehicle_angular_velocity_groundtruth
vehicle_angular_velocity_groundtruth = ulog.get_dataset('vehicle_angular_velocity_groundtruth')
if vehicle_angular_velocity_groundtruth:
    print(vehicle_angular_velocity_groundtruth.data)

vehicle_rates_setpoint = ulog.get_dataset('vehicle_rates_setpoint')
if vehicle_rates_setpoint:
    print(vehicle_rates_setpoint.data)

vehicle_local_position = ulog.get_dataset('vehicle_local_position')
if vehicle_local_position:
    print(vehicle_local_position.data)

vehicle_attitude_groundtruth = ulog.get_dataset('vehicle_attitude_groundtruth')
if vehicle_attitude_groundtruth:
    print(vehicle_attitude_groundtruth.data)

"""
请你根据我的代码和输出，编写这样一份代码：从ulog文件中读取轨迹，然后保存到一个json中作为数据集。要求如下：

1. 从10s后才开始采样，抛弃10s前的结果
2. 最终的数据集包含如下信息：时间戳，位置xyz，速度vx vy vz，姿态角
"""