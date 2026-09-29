# SmellEngine

Système d'olfaction artificielle en temps réel pour jeux vidéo, films et vidéos.
Capture d'écran → vision par ordinateur (YOLO-World + CLIP) → diffusion d'arômes via ESP32.

## À propos

SmellEngine analyse en temps réel le contenu affiché à l'écran (jeu vidéo, film, vidéo) et déclenche la diffusion d'odeurs correspondant à la scène détectée.

Le système combine :
- Capture d'écran native (macOS AppKit / Quartz)
- Détection d'objets via YOLO-World
- Compréhension de scène via CLIP
- Inférence optimisée avec ONNX Runtime
- Pilotage de 6 cartouches d'odeurs via ESP32
- Interface desktop native (Tauri + React)

Projet développé dans le cadre du Master 1 IAFA à l'Université Paul Sabatier (Toulouse).

## Stack technique

- Vision / IA : Python, PyTorch, YOLO-World, CLIP, ONNX Runtime, Hugging Face, CUDA
- Backend : FastAPI, Uvicorn, WebSockets, asyncio
- Frontend : Tauri, React, TypeScript, Vite, Tailwind CSS
- Capture écran : macOS AppKit, Quartz (PyObjC)
- Matériel : ESP32, 6 cartouches d'odeurs, WebSockets

## Installation

### Prérequis
- macOS
- Python 3.11
- Node.js 18+
- Rust (pour Tauri) — https://rustup.rs

### Backend
    python3.11 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

### Frontend
    npm install

## Lancement

### Backend
    cd backend
    source ../venv/bin/activate
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

### Frontend (dev)
    npm run dev

### Application desktop
    npm run tauri dev

## API Backend

| Méthode | Endpoint | Description |
|---|---|---|
| GET | /health | État des modèles |
| POST | /start | Démarre la boucle d'analyse |
| POST | /stop | Arrête la boucle |
| GET | /state | État courant |
| GET | /preview | Image avec bounding boxes |
| GET | /windows | Liste des fenêtres |
| WS | /ws | Stream temps réel |

## Fonctionnalités

- Capture d'écran multi-fenêtres
- Détection d'objets temps réel (YOLO-World)
- Classification de scène (CLIP)
- Inférence ONNX Runtime
- Mapping visuel → odeur sur 6 cartouches : pine, earth, ocean, smoke, rain, asphalt
- WebSocket bidirectionnel
- Interface desktop Tauri + React
- Firmware ESP32 (en cours)

## Roadmap

- [x] Pipeline YOLO-World + CLIP
- [x] Backend FastAPI + WebSocket
- [x] Capture écran native macOS
- [x] Interface Tauri/React
- [ ] Firmware ESP32
- [ ] Support Windows / Linux

## Auteur

**Daniil Zhdanov**
Master 1 IAFA — Université Paul Sabatier (Toulouse)

- GitHub : https://github.com/yeezysmem
- LinkedIn : https://www.linkedin.com/in/zhdanov-dan/
- Email : d.zhdanov.dev@gmail.com

## Licence

MIT