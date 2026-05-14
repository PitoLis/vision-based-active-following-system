# Vision-Based Active Following System

基于机器视觉的主动跟随系统 — 使用 AI 摄像头实时检测人体目标，驱动二维云台锁定追踪 + 轮式底盘自主跟随。

**Demo Video**: [Bilibili](https://www.bilibili.com/video/BV1QS9XBMEZ8/?share_source=copy_web&vd_source=3df518f6888f0a9f2e1679324c10ea46)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    MaixCam AI Camera                     │
│   YOLO人体检测 + ByteTracker多目标跟踪 → UART输出坐标     │
└────────────────┬───────────────────┬────────────────────┘
                 │ (X, Y, W, H)      │ (offset_x)
                 ▼                   ▼
┌────────────────────────┐  ┌─────────────────────────────┐
│   STM32F407 云台控制    │  │    STM32F407 底盘控制         │
│   • CAN总线多电机同步   │  │    • PID速度/位置闭环        │
│   • 二维舵机/步进追踪   │  │    • 超声波避障              │
│   • 位置/速度双模式     │  │    • 循迹 + 蓝牙/WiFi遥控    │
└────────────────────────┘  └─────────────────────────────┘
```

视觉模块检测到人体后，通过 UART 将目标坐标发送给两台 STM32：
- **云台**：根据目标偏移量实时调整角度，保持摄像头锁定目标
- **底盘**：根据目标距离和方向，自主跟随或避障

---

## 硬件清单

| 模块 | 型号/方案 | 用途 |
|------|-----------|------|
| AI 摄像头 | Sipeed MaixCam (M4ndFV2) | YOLO 目标检测 + ByteTracker 跟踪 |
| 主控 ×2 | STM32F407VET6 | 云台控制 / 底盘控制 |
| 电机驱动 | L298N / TB6612 | 直流电机 PWM 驱动 |
| 步进电机 | 42 步进 + A4988 驱动 | 云台二维转动 |
| 姿态传感器 | MPU6050 | 底盘姿态反馈 |
| 超声波 | HC-SR04 | 前方障碍物检测 |
| OLED | 0.96'' I2C | 实时状态显示 |
| 通信 | CAN 总线 + UART + HC-05 蓝牙 / ESP8266 WiFi | 模块间 / 远程遥控 |

---

## 软件技术栈

| 层级 | 技术 |
|------|------|
| 视觉推理 | YOLOv5 / YOLO11 (Maix NN) + ByteTracker |
| 云台固件 | STM32 HAL + CAN 多电机同步 + 位置/速度双模式 |
| 底盘固件 | STM32 HAL + PID 闭环 + 超声波避障 + MPU6050 姿态 |
| 通信协议 | UART (视觉→STM32) + CAN bus (多电机) + BT/WiFi (遥控) |

---

## 代码结构

```
src/
├── tracking parts/
│   ├── vision/
│   │   └── NO.1vision.py        # 主视觉程序：YOLO检测 + 跟踪 + UART输出
│   └── PTZ/
│       ├── location mode/       # 云台位置模式固件（CAN多电机同步）
│       └── speed mode/          # 云台速度模式固件
│
└── moving parts/
    └── car/
        ├── Core/Src/            # main.c, adc, gpio, tim, usart 驱动
        ├── Core/Inc/            # 对应头文件
        ├── HARDWARE/            # 外设驱动层
        │   ├── motor.c          # 直流电机驱动
        │   ├── pid.c            # PID 控制器
        │   ├── HC_SR04.c        # 超声波测距
        │   ├── MPU6050.c        # 六轴姿态传感器
        │   ├── cJSON.c          # JSON 解析
        │   └── oled.c           # OLED 显示屏
        ├── Drivers/             # STM32F4 HAL & CMSIS
        └── MDK-ARM/             # Keil uVision5 工程文件
```

---

## 视觉检测流程

1. **YOLO 推理** — MaixCam 运行 YOLOv5/YOLO11，检测 `class_id=0`（人体）
2. **ByteTracker 跟踪** — 给每个检测框分配 ID，处理遮挡和目标丢失
3. **坐标映射** — 计算目标中心点相对于画面中心的偏移量
4. **UART 下发** — 将 `(center_x, center_y, width, height)` 发送给 STM32

关键可调参数（`NO.1vision.py` 顶部）：
```python
conf_threshold = 0.35    # 检测置信度阈值
iou_threshold  = 0.45    # NMS IoU 阈值
max_lost_buff  = 80      # 目标丢失缓冲帧数
track_thresh   = 0.4     # 跟踪器初始化阈值
```

---

## 快速开始

### 1. 视觉端（MaixCam）

```bash
# 将 NO.1vision.py 上传至 MaixCam
# 通过 MaixVision IDE 或 SD 卡部署
# 摄像头分辨率会自动打印，记住去更新 STM32 的 CENTER_X / CENTER_Y
```

### 2. 固件端（STM32）

```bash
# 分别将云台和底盘的 Keil 工程编译烧录
# 云台: src/tracking parts/PTZ/location mode/MDK-ARM/
# 底盘: src/moving parts/car/MDK-ARM/
```

### 3. 接线

```
MaixCam UART TX  → STM32 UART RX (云台 + 底盘并联)
云台 CAN_H/CAN_L → 步进电机驱动板
底盘 PWM         → 直流电机驱动板
HC-SR04 Trig/Echo → STM32 GPIO
MPU6050 SDA/SCL  → STM32 I2C
```

---

## 技术亮点

- **AI + 传统控制融合**：YOLO 深度模型处理感知，PID + CAN 总线处理执行，各司其职
- **CAN 多电机同步**：云台两个步进电机通过 CAN 总线实现同步位置/速度控制，避免 UART 指令延迟
- **双模式云台**：位置模式（精确指向）和速度模式（平滑追踪）可切换，适应不同场景
- **完整避障逻辑**：超声波 + MPU6050 姿态补偿，跟随过程中检测到障碍物自动停车
- **模块化硬件驱动**：`HARDWARE/` 下每个外设独立封装，方便移植到其他 STM32 平台

---

## Star History

<div align="center">
<a href="https://star-history.com/#PitoLis/vision-based-active-following-system&Date">
<img src="https://api.star-history.com/svg?repos=PitoLis/vision-based-active-following-system&type=Date" width="500">
</a>
</div>

## License

MIT
