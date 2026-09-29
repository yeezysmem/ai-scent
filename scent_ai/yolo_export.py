from ultralytics import YOLOWorld

SCENT_CLASSES = [
    "tree", "pine tree", "conifer tree", "plant", "forest",
    "sea", "ocean", "lake", "river", "water", "wave",
    "fire", "smoke", "flame", "bonfire", "exhaust smoke",
    "dirt", "soil", "mud", "sand", "grass", "rock", "ground",
    "cloud", "fog", "puddle", "rain",
    "road", "asphalt", "street", "car", "truck", "pavement",
]

model = YOLOWorld("yolov8s-world.pt")
model.set_classes(SCENT_CLASSES)

# Змінено imgsz на 672 та додано opset=12
model.export(
    format="onnx",
    imgsz=672,
    opset=12,
    simplify=True,
)
print("✅ Модель успішно експортовано в yolov8s-world.onnx")