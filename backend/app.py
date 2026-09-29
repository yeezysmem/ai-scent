from __future__ import annotations
import sys
import asyncio
import colorsys
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
import socket
from threading import Lock
import time
import traceback
from typing import Any
import uvicorn

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.window_manager import MacOSWindowManager
from scent_ai.capture import ScreenCapture
from scent_ai.clip_classifier import ClipClassifier
from scent_ai.runtime import RuntimeSmellEngine
from scent_ai.segmentation_onnx import SegmentationModel
from scent_ai.yolo_world import YOLOWorldDetector

# Автоматично знаходимо папку models відносно цього файлу
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "segformer_onnx"
CLIP_DIR = BASE_DIR / "models" / "clip_onnx"
YOLO_ONNX_PATH = BASE_DIR / "models" / "yolov8s-world.onnx"


if getattr(sys, 'frozen', False):
    # Шлях, коли додаток скомпільовано в бінарник
    BASE_DIR = Path(sys.executable).parent
else:
    # Шлях під час звичайного запуску python app.py
    BASE_DIR = Path(__file__).parent.parent
    
# Функція для отримання локальної IP
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
print(f"🖥️  Local IP: {LOCAL_IP}")
print(f"📱 Access from phone: http://{LOCAL_IP}:8000")

window_manager = MacOSWindowManager()

segmenter: SegmentationModel | None = None
classifier: ClipClassifier | None = None
detector: YOLOWorldDetector | None = None

runtime: RuntimeSmellEngine | None = None
capture: ScreenCapture | None = None
latest_frame: Image.Image | None = None
selected_window_id: int | None = None
latest_frame_lock = Lock()

CAPTURE_INTERVAL_SECONDS = 5

latest_result: dict[str, Any] | None = None
latest_result_lock = Lock()
is_running: bool = False
capture_task: asyncio.Task | None = None


# ===== WebSocket Manager (визначаємо до app) =====
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


async def _capture_loop() -> None:
    global latest_frame, latest_result, is_running  # додано `is_running`, щоб уникнути UnboundLocalError

    while is_running:
        try:
            screen_capture = get_capture()
            runtime_engine = get_runtime()

            loop_start = time.perf_counter()

            # --- 1. CAPTURE ---
            capture_start = time.perf_counter()

            # Якщо вибране конкретне вікно, передаємо selected_window_id у capture
            if selected_window_id is not None:
                image = await asyncio.to_thread(
                    screen_capture.capture, window_id=selected_window_id
                )
            else:
                image = await asyncio.to_thread(screen_capture.capture)

            capture_time = (time.perf_counter() - capture_start) * 1000

            # Збереження кадру для прев'ю
            with latest_frame_lock:
                latest_frame = image.copy()

            # --- 2. PROCESSING (YOLO + CLIP) ---
            process_start = time.perf_counter()

            # Обробка об'єктів та сцени у фоновому потоці, щоб не блокувати Event Loop
            result = await asyncio.to_thread(
                runtime_engine.process_with_yolo_clip, image, force=True
            )

            process_time = (time.perf_counter() - process_start) * 1000
            total_time = (time.perf_counter() - loop_start) * 1000

            # --- 3. STATE UPDATE ---
            analysis_data = result.get("analysis", {})
            yolo_data = analysis_data.get("yolo_results", {})

            with latest_result_lock:
                latest_result = {
                    "status": "ok",
                    "screenshot_size": {"width": image.width, "height": image.height},
                    "cartridges": runtime_engine.get_state(),  # [0.0 ... 1.0]
                    "percentages": runtime_engine.get_percent_state(),  # {pine: 96, ...}
                    "pwm": runtime_engine.get_pwm_state(),  # {pine: 245, ...}
                    "sources": analysis_data.get("final_sources", {}),
                    "scene_areas": yolo_data.get("percentages", {}),
                    "object_analyses": analysis_data.get("object_analyses", []),
                    "should_send": result.get("should_send", False),
                    "metrics": {
                        "capture_time_ms": round(capture_time, 2),
                        "process_time_ms": round(process_time, 2),
                        "segmentation_time_ms": 0,
                        "classification_time_ms": round(process_time * 0.6, 2),
                        "detection_time_ms": round(process_time * 0.4, 2),
                        "total_time_ms": round(total_time, 2),
                        "fps": round(1000 / total_time, 1) if total_time > 0 else 0,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    },
                }

            # Розсилка по WebSocket
            await manager.broadcast(latest_result)

        except asyncio.CancelledError:
            break
        except Exception as error:
            print(f"❌ Capture loop error: {error}")
            traceback.print_exc()

        await asyncio.sleep(CAPTURE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, classifier, runtime, capture

    print("Loading models...")
    model_path = MODELS_DIR
    # segmenter = SegmentationModel()
    onnx_path_str = str(YOLO_ONNX_PATH)
    detector = YOLOWorldDetector(model_path=onnx_path_str, confidence_threshold=0.25)
    print("Yolo-world loaded")
    classifier = ClipClassifier()
    print("Clip model loaded")

    runtime = RuntimeSmellEngine(
        segmenter=None,
        classifier=classifier,
        detector=detector,
        activation_threshold=0.01,
        smoothing_alpha=0.5,
        change_threshold=0.05,
        update_interval=1.0,
        max_active_cartridges=3,
    )

    capture = ScreenCapture(monitor_index=1, window_manager=window_manager)
    print("AI Scent backend ready.")

    yield

    global is_running
    is_running = False
    if capture_task is not None:
        capture_task.cancel()

    if capture is not None:
        capture.close()


# ===== Функції get_runtime та get_capture (визначаємо до app) =====
def get_runtime() -> RuntimeSmellEngine:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Runtime engine is not ready.")
    return runtime


def get_capture() -> ScreenCapture:
    if capture is None:
        raise HTTPException(status_code=503, detail="Screen capture is not ready.")
    return capture


# ===== Створюємо FastAPI додаток (ПІСЛЯ визначення lifespan) =====
app = FastAPI(title="AI Scent Engine", version="0.1.0", lifespan=lifespan)

# ===== CORS налаштування =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        f"http://{LOCAL_IP}:5173",
        f"http://{LOCAL_IP}:3000",
        "http://192.168.1.*",
        "http://192.168.0.*",
        "http://10.0.0.*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== WebSocket ендпоінт =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            with latest_result_lock:
                if latest_result:
                    await websocket.send_json(latest_result)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ===== Ендпоінти API =====
@app.get("/ip")
def get_ip():
    return {"ip": LOCAL_IP}


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "AI Scent Backend running"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": (detector is not None and classifier is not None and runtime is not None),
        "capture_ready": capture is not None,
        "detector_ready": detector is not None,
        "classifier_ready": classifier is not None,
    }


@app.post("/analyze")
def analyze_screen() -> dict[str, Any]:
    runtime_engine = get_runtime()
    screen_capture = get_capture()

    try:
        # ✅ Використовуємо selected_window_id, якщо воно обрано
        if selected_window_id is not None:
            image = screen_capture.capture(window_id=selected_window_id)
        else:
            image = screen_capture.capture()

        global latest_frame

        with latest_frame_lock:
            latest_frame = image.copy()

        result = runtime_engine.process(image, force=True)
        analysis = result["analysis"]

        return {
            "status": "ok",
            "screenshot_size": {"width": image.width, "height": image.height},
            "cartridges": result["cartridges"],
            "percentages": runtime_engine.get_percent_state(),
            "pwm": runtime_engine.get_pwm_state(),
            "sources": analysis["final_sources"],
            "scene_areas": analysis["scene_areas"],
            "should_send": result["should_send"],
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/start")
async def start_analysis() -> dict[str, Any]:
    global is_running, capture_task

    get_runtime()
    get_capture()

    if is_running:
        return {"status": "already_running"}

    is_running = True
    capture_task = asyncio.create_task(_capture_loop())
    return {"status": "started", "interval_seconds": CAPTURE_INTERVAL_SECONDS}


@app.get("/state")
def current_state() -> dict[str, Any]:
    runtime_engine = get_runtime()

    with latest_result_lock:
        cached = dict(latest_result) if latest_result is not None else None

    return {
        "status": "ok",
        "is_running": is_running,
        "cartridges": runtime_engine.get_state(),
        "percentages": runtime_engine.get_percent_state(),
        "pwm": runtime_engine.get_pwm_state(),
        "latest": cached,
    }


@app.get("/preview")
def preview():
    """Повертає preview з візуалізацією YOLO-World + CLIP + CARTRIDGES"""
    with latest_frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=404, detail="No captured frame available yet.")
        frame = latest_frame.copy()

    with latest_result_lock:
        result = dict(latest_result) if latest_result is not None else None

    # ===== ВІЗУАЛІЗАЦІЯ YOLO-WORLD =====
    if detector is not None:
        try:
            # Отримуємо детекцію
            detection = detector.detect(frame)

            # Створюємо копію зображення для малювання
            preview_image = frame.copy()
            draw = ImageDraw.Draw(preview_image)

            # Кольори для різних класів
            def get_color(class_name):
                """Генерує колір для класу"""
                hue = hash(class_name) % 100 / 100
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
                return tuple(int(c * 255) for c in rgb)

            # Малюємо bounding boxes
            objects = detection.get("objects", {})
            for class_name, boxes in objects.items():
                color = get_color(class_name)
                for box in boxes:
                    bbox = box["bbox"]
                    confidence = box["confidence"]

                    # Малюємо прямокутник
                    draw.rectangle(bbox, outline=color, width=3)

                    # Підпис з назвою класу та впевненістю
                    label = f"{class_name} {confidence:.2f}"
                    draw.text((bbox[0], bbox[1] - 20), label, fill=color)

            # Додаємо статистику внизу
            text_area_height = 300
            h, w = frame.size[1], frame.size[0]
            new_height = h + text_area_height

            # Створюємо нове зображення з місцем для тексту
            new_image = Image.new("RGB", (w, new_height), color=(15, 15, 35))
            new_image.paste(preview_image, (0, 0))

            draw = ImageDraw.Draw(new_image)

            # Завантажуємо шрифти
            try:
                font_paths = [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/System/Library/Fonts/HelveticaNeue.ttc",
                    "/Library/Fonts/Arial.ttf",
                ]
                font = None
                font_bold = None
                for path in font_paths:
                    try:
                        font = ImageFont.truetype(path, 18)
                        font_bold = ImageFont.truetype(path, 20)
                        break
                    except Exception:
                        continue
                if font is None:
                    raise Exception("No font found")
            except Exception:
                font = ImageFont.load_default()
                font_bold = font

            y_offset = h + 15
            x_offset = 20
            line_height = 24

            # ===== 1. YOLO-WORLD STATS =====
            draw.text(
                (x_offset, y_offset),
                "🎯 YOLO-WORLD DETECTIONS",
                fill=(255, 255, 255),
                font=font_bold,
            )
            y_offset += line_height + 5

            percentages = detection.get("percentages", {})
            sorted_classes = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

            if sorted_classes:
                # Розбиваємо на 2 колонки
                half = (len(sorted_classes) + 1) // 2
                col1 = sorted_classes[:half]
                col2 = sorted_classes[half:]

                # Перша колонка
                x1 = x_offset
                y1 = y_offset

                for class_name, pct in col1:
                    color = get_color(class_name)
                    bar_width = int(pct * 300)

                    # Квадратик кольору
                    draw.rectangle([x1, y1, x1 + 14, y1 + 14], fill=color)

                    # Назва та відсоток
                    text = f"{class_name}: {pct*100:.1f}%"
                    draw.text((x1 + 20, y1 - 2), text, fill=(255, 255, 255), font=font)

                    # Смужка
                    draw.rectangle([x1 + 200, y1 + 4, x1 + 200 + bar_width, y1 + 14], fill=color)

                    y1 += line_height

                # Друга колонка
                if col2:
                    x2 = x_offset + (w // 2)
                    y2 = y_offset

                    for class_name, pct in col2:
                        color = get_color(class_name)
                        bar_width = int(pct * 300)

                        draw.rectangle([x2, y2, x2 + 14, y2 + 14], fill=color)
                        text = f"{class_name}: {pct*100:.1f}%"
                        draw.text((x2 + 20, y2 - 2), text, fill=(255, 255, 255), font=font)
                        draw.rectangle([x2 + 200, y2 + 4, x2 + 200 + bar_width, y2 + 14], fill=color)

                        y2 += line_height

                max_y = max(y1 if col1 else y_offset, y2 if col2 else y_offset) + 10
            else:
                draw.text(
                    (x_offset, y_offset),
                    "No objects detected",
                    fill=(255, 255, 255),
                    font=font,
                )
                max_y = y_offset + line_height + 10

            # ===== 2. CLIP =====
            draw.text((x_offset, max_y), "🧠 CLIP", fill=(255, 255, 255), font=font_bold)
            max_y += line_height + 5

            if result and result.get("sources"):
                sources = result.get("sources", {})
                if sources and isinstance(sources, dict):
                    sorted_sources = sorted(
                        sources.items(),
                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                        reverse=True,
                    )[:4]
                    line = ""
                    for source_name, score in sorted_sources:
                        if isinstance(score, (int, float)) and score > 0.05:
                            line += f"  {source_name}: {score:.2f}  "
                    draw.text((x_offset, max_y), line, fill=(200, 255, 200), font=font)
                    max_y += line_height + 5

            # ===== 3. CARTRIDGES =====
            draw.text((x_offset, max_y), "💨 CARTRIDGES", fill=(255, 255, 255), font=font_bold)
            max_y += line_height + 5

            if result and result.get("cartridges"):
                cartridges = result.get("cartridges", {})
                if cartridges and isinstance(cartridges, dict):
                    # Кольори для картриджів
                    cartridge_colors = {
                        "pine": (34, 139, 34),
                        "earth": (139, 69, 19),
                        "ocean": (0, 119, 190),
                        "smoke": (169, 169, 169),
                        "rain": (70, 130, 180),
                        "asphalt": (105, 105, 105),
                    }

                    line = ""
                    for name, value in cartridges.items():
                        if isinstance(value, (int, float)) and value > 0.05:
                            color = cartridge_colors.get(name, (200, 200, 200))
                            bar_width = int(value * 300)
                            line += f"  {name.upper()}: {int(value * 100)}%  "

                    draw.text((x_offset, max_y), line, fill=(255, 255, 200), font=font)

            preview_image = new_image

        except Exception as e:
            print(f"Preview error: {e}")
            traceback.print_exc()
            preview_image = frame
    else:
        preview_image = frame

    buffer = BytesIO()
    preview_image.save(buffer, format="JPEG", quality=85, optimize=True)

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/windows")
def get_windows():
    return {"windows": [window.to_dict() for window in window_manager.list_windows()]}


@app.post("/stop")
async def stop() -> dict[str, Any]:
    global is_running, capture_task

    is_running = False

    if capture_task is not None:
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass
        capture_task = None

    runtime_engine = get_runtime()
    stopped_state = runtime_engine.stop()
    return {"status": "stopped", "cartridges": stopped_state}


# Додайте цю функцію в app.py
@app.post("/force-update")
async def force_update():
    """Примусово відправляє останні дані через WebSocket"""
    with latest_result_lock:
        if latest_result:
            await manager.broadcast(latest_result)
            return {"status": "ok", "message": "Data sent"}
    return {"status": "error", "message": "No data"}


@app.get("/segmentation")
def get_segmentation():
    """Повертає візуалізацію сегментації"""
    with latest_frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=404, detail="No captured frame available yet.")
        frame = latest_frame.copy()

    # Отримуємо сегментацію
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")

    result = segmenter.predict(frame)

    # Створюємо кольорову маску
    colored_mask = segmenter.visualize(result, frame)

    # Зберігаємо в буфер
    buffer = BytesIO()
    colored_mask.save(buffer, format="JPEG", quality=85)

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/segmentation-viz")
def segmentation_viz():
    """Візуалізація сегментації"""
    global segmenter, latest_frame

    if segmenter is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")

    with latest_frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=404, detail="No captured frame available yet.")
        frame = latest_frame.copy()

    # Отримуємо маску
    mask = segmenter.predict(frame)

    # Створюємо візуалізацію
    viz = segmenter.visualize(mask, frame)

    # Зберігаємо в буфер
    buffer = BytesIO()
    viz.save(buffer, format="JPEG", quality=85)

    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/windows/select/{window_id}")
def select_window(window_id: int):
    global selected_window_id
    if window_id == 0:
        selected_window_id = None
        return {"ok": True, "selected": "Full Screen"}

    if not window_manager.window_exists(window_id):
        raise HTTPException(404, "Window not found")
    selected_window_id = window_id
    return {"ok": True, "selected_window_id": selected_window_id}

 
if __name__ == "__main__":
    # Запускаємо сервер на порту 8000 без reload
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)