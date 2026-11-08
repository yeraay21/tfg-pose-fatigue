"""
Separa las repeticiones de sentadilla a partir de la curva de rodilla de angles/.
Cada mínimo de rodilla (pico de −rodilla) es una repetición; va de pico de pie a pico de pie.
Salida: reps.csv con una fila por repetición.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Configuración
METADATA = Path("dataset/metadata.csv")
DATASET = Path("dataset")
ANGLES_DIR = Path("angles")
OUT_CSV = Path("reps.csv")
EXERCISE = "squats"
PROMINENCE = 30       # grados que un valle debe destacar sobre lo que le rodea
DISTANCE_S = 1.0      # separación mínima entre dos valles, en segundos
WINDOW_S = 0.2        # zona alrededor del extremo para NaN y saltos (±3 frames a 15 fps)
INTERP_WINDOW = 1     # frames a cada lado del extremo donde un frame interpolado invalida la métrica
# Cambio por encima de esto = salto. ≈1.3 × velocidad máxima del sujeto más limpio (018)
JUMP_DEG_S = {"knee": 450, "hip": 500, "torso": 200}


def get_fps(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def find_reps(knee, fps):
    """Devuelve (mins, tops): frames de los mínimos y de los picos de pie que los rodean.
    La repetición i va de tops[i] a tops[i + 1] con mínimo en mins[i]."""
    # find_peaks no admite NaN: se rellena SOLO para detectar, las métricas siguen con sus NaN
    filled = knee.interpolate(limit_direction="both").to_numpy()
    mins, _ = find_peaks(-filled, prominence=PROMINENCE, distance=max(1, round(DISTANCE_S * fps)))
    # Pico de pie = máximo de rodilla entre dos mínimos consecutivos (o hasta el borde del vídeo)
    bounds = [0] + mins.tolist() + [len(filled) - 1]
    tops = [a + int(np.argmax(filled[a:b + 1])) for a, b in zip(bounds[:-1], bounds[1:])]
    return mins, tops


def is_valid(angles, metric, frame, fps):
    """False si cerca del extremo de la métrica hay NaN, un salto o un frame interpolado (ventana más estrecha)."""
    w = round(WINDOW_S * fps)
    low, high = max(0, frame - w), frame + w + 1
    if angles[metric].iloc[low:high].isna().any():
        return False
    near = angles[f"interp_{metric}"].iloc[max(0, frame - INTERP_WINDOW):frame + INTERP_WINDOW + 1]
    if near.any():
        return False
    # diff sobre la serie entera para no perder el cambio que entra en la ventana
    speed = angles[metric].diff().abs() * fps
    return not (speed.iloc[low:high] > JUMP_DEG_S[metric]).any()


def rep_rows(row, angles, fps):
    """Una fila (dict) por repetición del vídeo."""
    mins, tops = find_reps(angles["knee"], fps)
    last_frame = angles["knee"].last_valid_index()
    rows = []
    for i, frame_min in enumerate(mins):
        start, end = tops[i], tops[i + 1]
        rep = angles.iloc[start:end + 1]
        # Si el pico cae en el borde, el vídeo empezó o acabó a medias: esa fase no se vio entera
        cut_start, cut_end = start == 0, end >= last_frame
        # Cada métrica se valida alrededor de su propio extremo dentro de la repetición
        extremes = {"knee": frame_min, "hip": rep["hip"].idxmin(), "torso": rep["torso"].idxmax()}
        rows.append({
            "subject": row.subject,
            "label": row.label,
            "rep": i + 1,
            "frame_inicio": start,
            "frame_min": int(frame_min),
            "frame_fin": end,
            "knee_min": angles["knee"].iloc[frame_min],
            "hip_min": rep["hip"].min(),
            "torso_max": rep["torso"].max(),
            "t_bajada_s": np.nan if cut_start else (frame_min - start) / fps,
            "t_subida_s": np.nan if cut_end else (end - frame_min) / fps,
            **{f"valid_{m}": is_valid(angles, m, int(f), fps) for m, f in extremes.items()},
            "partial": cut_start or cut_end,
        })
    return rows


def main():
    # keep_default_na=False: las celdas vacías se leen como "" en vez de NaN
    meta = pd.read_csv(METADATA, dtype=str, keep_default_na=False)
    todo = meta[(meta["subject_ids"] != "") & (meta["exercise"] == EXERCISE)]

    all_rows = []
    for row in todo.itertuples():
        stem = Path(row.file).stem
        fps = get_fps(DATASET / row.exercise / row.label / row.file)
        rows = rep_rows(row, pd.read_csv(ANGLES_DIR / f"{stem}.csv"), fps)
        all_rows.extend(rows)
        invalid = {m: [r["rep"] for r in rows if not r[f"valid_{m}"]] for m in JUMP_DEG_S}
        print(f"{stem}: fps {fps:.0f} | reps {len(rows)} | no válidas {invalid}")

    reps = pd.DataFrame(all_rows).round(3)
    reps.to_csv(OUT_CSV, index=False)
    valid = {m: int(reps[f"valid_{m}"].sum()) for m in JUMP_DEG_S}
    print(f"\n{len(reps)} repeticiones en {OUT_CSV} | válidas por métrica: {valid}")


if __name__ == "__main__":
    main()
