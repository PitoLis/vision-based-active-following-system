import time  
from maix import nn, display, camera, app, image, tracker, sys, uart

colors = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], 
    [255, 255, 0], [0, 255, 255], [255, 0, 255],
]

def show_tracks(img : image.Image, tracks):
    valid = 0
    for track in tracks:
        if track.lost: continue
        valid += 1
        color = colors[track.id % len(colors)]
        color = image.Color.from_rgb(color[0], color[1], color[2])
        obj = track.history[-1]
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color, thickness=2)
        img.draw_string(obj.x, obj.y-15, f"ID:{track.id}", color, scale=1.2)
    img.draw_string(2, 2, f'Track: {valid}', image.COLOR_RED, scale=1.2)

# ===================== 全局参数配置区 =====================
conf_threshold = 0.35 
iou_threshold = 0.45 
max_lost_buff_time = 80 
track_thresh = 0.4
high_thresh = 0.6
match_thresh = 0.8
valid_class_id = 0 
# ==========================================================

# ===================== 初始化硬件 =====================
if sys.device_name().lower() == "maixcam2":
    detector = nn.YOLO11(model="/root/models/yolo11s.mud", dual_buff=True)
else:
    detector = nn.YOLOv5(model="/root/models/yolov5s.mud", dual_buff=True)

disp = display.Display()
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
tracker0 = tracker.ByteTracker(max_lost_buff_time, track_thresh, high_thresh, match_thresh)

devices = uart.list_devices()
serial = uart.UART(devices[0], 115200) 
# ======================================================

last_send_time = 0  

# 获取实际分辨率，并打印出来提醒你改 STM32
cam_w = detector.input_width()
cam_h = detector.input_height()
print(f"==================================================")
print(f"⚠️ 警告: 当前摄像头真实分辨率为 {cam_w} x {cam_h}")
print(f"⚠️ 请立刻将 STM32 代码里的 CENTER_X 改为 {cam_w//2}，CENTER_Y 改为 {cam_h//2}")
print(f"==================================================")

while not app.need_exit():
    img = cam.read()
    objs = detector.detect(img, conf_th=conf_threshold, iou_th=iou_threshold)

    person_objs = [obj for obj in objs if obj.class_id == valid_class_id]

    max_area = 0
    max_person = None
    for p in person_objs:
        area = p.w * p.h
        if area > max_area:
            max_area = area
            max_person = p

    tracker_objs = []
    if max_person:
        tracker_objs.append(tracker.Object(
            max_person.x, max_person.y,
            max_person.w, max_person.h,
            max_person.class_id, max_person.score
        ))

    # 喂给追踪器，获取平滑后的轨迹
    tracks = tracker0.update(tracker_objs)

    target_x = None
    target_y = None

    # ===================== ✅ 核心修复：使用 ByteTrack 平滑后的数据 =====================
    for track in tracks:
        if track.lost:
            continue
        
        # 拿到卡尔曼滤波平滑后的最新坐标
        smooth_obj = track.history[-1] 
        target_x = smooth_obj.x + smooth_obj.w // 2
        target_y = smooth_obj.y + smooth_obj.h // 2

        img.draw_circle(target_x, target_y, 5, image.COLOR_RED, thickness=-1)
        img.draw_string(2, 25, f"X:{target_x} Y:{target_y}", image.COLOR_GREEN, scale=1.2)

        # 只要找到了平滑后的目标，就跳出循环（保证只发最大的那个）
        break 

    # 节流阀：0.05 秒发送一次 (20Hz)
    current_time = time.time()  
    if target_x is not None and target_y is not None:
        if current_time - last_send_time >= 0.05:  
            last_send_time = current_time         
            data = f"X{target_x}Y{target_y}\n"
            serial.write_str(data)   
            #print("发送：", data) # 调试好之后建议注释掉，防刷屏

    show_tracks(img, tracks)
    disp.show(img)