"""
Extrae los keypoints de YOLO-Pose frame a frame y los guarda en un CSV por vídeo.
Procesa los vídeos de metadata.csv que tengan rellena la columna subject_ids
(IDs de tracking del sujeto separados por ';', elegidos a mano con inspect_tracks.py).
Columnas: frame, x0, y0, c0, ..., x16, y16, c16
Si en un frame no está el sujeto -> fila con NaN (no saltarse frames).
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

# Configuración
METADATA = Path("dataset/metadata.csv")
DATASET = Path("dataset")
MODEL = "yolo26m-pose.pt"
TRACKER = "tracktrack.yaml"   # debe ser el mismo que en inspect_tracks.py
OUT_DIR = Path("keypoints")
N_KPTS = 17


def build_columns():
    columns = ["frame"]
    for i in range(N_KPTS):
        columns.extend([f"x{i}", f"y{i}", f"c{i}"])
    return columns


def parse_ids(text):
    """'1;21;43' -> [1, 21, 43]"""
    return [int(t) for t in text.split(";") if t.strip()]


def extract(video, subject_ids):
    # Igual que en inspect_tracks.py: modelo nuevo + contador de IDs a cero,
    # para que los IDs coincidan con los que se apuntaron a mano
    model = YOLO(MODEL)
    BaseTrack.reset_id()
    rows = []

    results = model.track(str(video), stream=True, verbose=False, persist=True, tracker=TRACKER)

    for frame_idx, result in enumerate(results):
        track_ids = result.boxes.id.cpu().numpy().astype(int).tolist() if result.boxes.id is not None else []

        # Primer ID del frame que pertenezca al sujeto (None si no está)
        subject_id = next((t for t in track_ids if t in subject_ids), None)

        if subject_id is None:
            row = [frame_idx] + [np.nan] * (N_KPTS * 3)
        else:
            pos = track_ids.index(subject_id)
            xy = result.keypoints.xy[pos].cpu().numpy()
            conf = result.keypoints.conf[pos].cpu().numpy()
            row = [frame_idx] + np.column_stack((xy, conf)).flatten().tolist()

        rows.append(row)

    return pd.DataFrame(rows, columns=build_columns())


def main():
    # keep_default_na=False: las celdas vacías se leen como "" en vez de NaN
    meta = pd.read_csv(METADATA, dtype=str, keep_default_na=False)
    todo = meta[meta["subject_ids"] != ""]
    OUT_DIR.mkdir(exist_ok=True)

    for r in todo.itertuples():
        video = DATASET / r.exercise / r.label / r.file
        df = extract(video, parse_ids(r.subject_ids))
        df.to_csv(OUT_DIR / f"{video.stem}.csv", index=False)

        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Confianza media de rodillas+tobillos por lado: ayuda a comprobar near_side
        conf_left = df[["c13", "c15"]].mean().mean()
        conf_right = df[["c14", "c16"]].mean().mean()
        print(f"{r.file}: filas {len(df)} | frames {n_frames} | sin sujeto {df['x0'].isna().sum()} "
              f"| conf izq {conf_left:.2f} der {conf_right:.2f} | near_side={r.near_side or '?'}")


if __name__ == "__main__":
    main()
