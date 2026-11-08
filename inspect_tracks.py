"""
Ayuda para elegir a mano el ID de tracking del sujeto en cada vídeo.
Para cada vídeo: ejecuta model.track, imprime por cada ID
(nº de frames, primer/último frame, área media de la bbox)
y guarda el vídeo anotado con los IDs y el número de frame.
"""
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

# Configuración
METADATA = Path("dataset/metadata.csv")
DATASET = Path("dataset")
MODEL = "yolo26m-pose.pt"
TRACKER = "tracktrack.yaml"   # explícito: el defecto de ultralytics puede cambiar entre versiones
EXERCISE = "squats"
SUBJECTS = ["004", "018", "020", "022"]
OUT_DIR = Path("runs/inspect_tracks") 


def video_paths():
    meta = pd.read_csv(METADATA, dtype=str)  # dtype=str: si no, "004" se lee como el número 4
    sel = meta[(meta["exercise"] == EXERCISE) & (meta["subject"].isin(SUBJECTS))]
    return [DATASET / r.exercise / r.label / r.file for r in sel.itertuples()]


def inspect(video):
    # Modelo nuevo por vídeo -> tracker nuevo (sin estado del vídeo anterior)
    model = YOLO(MODEL)
    # El contador de IDs es global y TrackTrack no lo reinicia -> lo hacemos a mano
    # para que los IDs empiecen en 1 en cada vídeo (independientes del orden de proceso)
    BaseTrack.reset_id()
    stats = {}  # id -> {"n": nº frames, "first": ..., "last": ..., "area_sum": ...}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    writer = None

    # mismos argumentos que en extract_keypoints.py, o los IDs no coincidirán
    results = model.track(str(video), stream=True, verbose=False, persist=True, tracker=TRACKER)

    for frame_idx, result in enumerate(results):
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)    # (personas,)
            wh = result.boxes.xywh[:, 2:].cpu().numpy()        # (personas, 2): ancho y alto en px
            for tid, (w, h) in zip(ids, wh):
                s = stats.setdefault(tid, {"n": 0, "first": frame_idx, "last": frame_idx, "area_sum": 0.0})
                s["n"] += 1
                s["last"] = frame_idx
                s["area_sum"] += w * h

        # Vídeo anotado
        img = result.plot()  # imagen BGR (array numpy) con bboxes, "id:N" y esqueleto
        # Nº de frame arriba a la izquierda (para ver a ojo en qué frame cambia el ID)
        cv2.putText(img, f"frame {frame_idx}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 255), 3)
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = OUT_DIR / f"{video.stem}_tracks.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        writer.write(img)

    writer.release()
    return stats


def print_stats(video, stats):
    print(f"\n{video.name}")
    print(f"{'ID':>4} | {'frames':>6} | {'primero-último':>14} | {'área media':>10}")
    for tid, s in sorted(stats.items(), key=lambda kv: kv[1]["n"], reverse=True):
        rango = f"{s['first']}-{s['last']}"
        print(f"{tid:>4} | {s['n']:>6} | {rango:>14} | {s['area_sum'] / s['n']:>10.0f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for video in video_paths():
        stats = inspect(video)
        print_stats(video, stats)


if __name__ == "__main__":
    main()
