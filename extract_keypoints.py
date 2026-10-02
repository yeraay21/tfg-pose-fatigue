"""
Extrae los keypoints de YOLO-Pose frame a frame de UN vídeo y los guarda en CSV.
Columnas: frame, x0, y0, c0, ..., x16, y16, c16
Si en un frame no está el sujeto -> fila con NaN (no saltarse frames).
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Configuración
VIDEO = "dataset/squats/correct/subject_022_squat_good_side.mp4"
MODEL = "yolo26m-pose.pt"
SUBJECT_IDX = 0          # índice del sujeto en la lista de detecciones (elegido a mano)
SUBJECT_ID = 1           # ID del tracker del sujeto
OUT_DIR = Path("keypoints")
N_KPTS = 17


def build_columns():
    columns = ["frame"] # frame index
    for i in range(N_KPTS): # 17 keypoints
        columns.extend([f"x{i}", f"y{i}", f"c{i}"])
    return columns


def main():
    model = YOLO(MODEL)
    rows = []

    results = model.track(VIDEO, stream=True, verbose=False, persist=True)  # track the video frame by frame
    
    for frame_idx, result in enumerate(results):

        # TODO: si result.boxes.id es None -> track_ids = []
        #       si no -> pasar a lista de enteros
        # Pista: .cpu().numpy().astype(int).tolist()
        track_ids = result.boxes.id.cpu().numpy().astype(int).tolist() if result.boxes.id is not None else []

        
        if SUBJECT_ID not in track_ids:
            row = [frame_idx] + [np.nan] * (N_KPTS * 3) 
        else:
            pos = track_ids.index(SUBJECT_ID)
            xy = result.keypoints.xy[pos].cpu().numpy()  # x and y coordinates
            conf = result.keypoints.conf[pos].cpu().numpy()  # confidence scores
            row = [frame_idx] + np.column_stack((xy, conf)).flatten().tolist()

        rows.append(row)

    df = pd.DataFrame(rows, columns=build_columns())
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / f"{Path(VIDEO).stem}.csv", index=False)

    cap = cv2.VideoCapture(VIDEO)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Filas: {len(df)} | Frames: {n_frames} | Sin sujeto: {df['x0'].isna().sum()}")

if __name__ == "__main__":
    main()
