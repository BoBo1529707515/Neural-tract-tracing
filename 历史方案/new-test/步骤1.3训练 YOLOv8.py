from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(
    data=r'F:\工作文件\RA\python\项目汇总\神经图像\yolo_dataset\data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    patience=20,
    project='neuro_detect',
    name='exp1'
)

print("\n" + "="*50)
print("✅ 训练完成！")
print(f"📂 最佳模型: neuro_detect/exp1/weights/best.pt")
