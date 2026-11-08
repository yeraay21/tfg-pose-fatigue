"""
Calcula las métricas articulares frame a frame, en lote, a partir de los CSV de keypoints.
Procesa las filas de metadata.csv con subject_ids del ejercicio EXERCISE y usa solo
el lado cercano a la cámara (columna near_side).
Ángulo interior en el vértice v: θ = arccos( (u−v)·(p−v) / (|u−v|·|p−v|) ), en 0-180°.
Salida: un CSV por vídeo en angles/ con columnas frame, knee, hip, torso e interp_<métrica>.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Configuración
METADATA = Path("dataset/metadata.csv")
KP_DIR = Path("keypoints")
OUT_DIR = Path("angles")
EXERCISE = "squats"      # valor de la columna exercise de metadata.csv
CONF_MIN = 0.5           # keypoints con menos confianza -> NaN
MAX_GAP = 5              # huecos de NaN de hasta MAX_GAP frames se interpolan

# Tripletas (u, v, p) por ejercicio y lado; v es el vértice del ángulo
TRIPLETS = {
    "squats": {
        "knee": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "hip": {"left": (5, 11, 13), "right": (6, 12, 14)},   # hombro-cadera-rodilla
    },
    "bicep_curl": {
        "elbow": {"left": (5, 7, 9), "right": (6, 8, 10)},
    },
}
# Keypoints (hombro, cadera) por lado, para la inclinación del torso
TORSO = {"left": (5, 11), "right": (6, 12)}


def fill_short_gaps(s):
    """Interpola solo los huecos de longitud total <= MAX_GAP; los largos quedan enteros en NaN.
    Ojo: interpolate(limit=N) no sirve, porque rellenaría los N primeros frames de un hueco largo."""
    isna = s.isna()
    # Cada valor válido abre un grupo con los NaN que le siguen -> suma = longitud del hueco
    gap_len = isna.groupby((~isna).cumsum()).transform("sum")
    return s.interpolate(limit_area="inside").where(~isna | (gap_len <= MAX_GAP))


def clean_keypoints(df):
    """Pone a NaN los keypoints poco fiables e interpola huecos cortos.
    Devuelve (df limpio, máscara booleana con las coordenadas que se han interpolado)."""
    df = df.copy()
    for i in range(17):
        mask = df[f"c{i}"] < CONF_MIN
        df.loc[mask, [f"x{i}", f"y{i}"]] = np.nan
    coords = [f"{axis}{i}" for i in range(17) for axis in "xy"]
    was_nan = df[coords].isna()
    df[coords] = df[coords].apply(fill_short_gaps)
    return df, was_nan & df[coords].notna()


def angle(df, u, v, p):
    """Ángulo interior en v (grados, 0-180) para todos los frames a la vez."""
    # Vectores desde el vértice v hacia u y hacia p, shape (frames, 2)
    a = df[[f"x{u}", f"y{u}"]].to_numpy() - df[[f"x{v}", f"y{v}"]].to_numpy()
    b = df[[f"x{p}", f"y{p}"]].to_numpy() - df[[f"x{v}", f"y{v}"]].to_numpy()
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    # clip: por redondeo cos puede salir 1.0000001 y arccos daría NaN
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def torso_angle(df, shoulder, hip):
    """Inclinación del torso SIN signo: ángulo entre (hombro − cadera) y la vertical (0, −1)."""
    a = df[[f"x{shoulder}", f"y{shoulder}"]].to_numpy() - df[[f"x{hip}", f"y{hip}"]].to_numpy()
    # Con b = (0, −1): a·b = −a_y y |b| = 1. Tronco vertical -> 0°, horizontal -> 90°
    cos = -a[:, 1] / np.linalg.norm(a, axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def compute_metrics(row):
    """Devuelve un DataFrame (frame, knee, hip, torso) para una fila de metadata."""
    df = pd.read_csv(KP_DIR / f"{Path(row.file).stem}.csv")
    df, filled = clean_keypoints(df)
    out = pd.DataFrame({"frame": df["frame"]})

    for name, sides in TRIPLETS[EXERCISE].items():
        u, v, p = sides[row.near_side]
        out[name] = angle(df, u, v, p)

    shoulder, hip = TORSO[row.near_side]
    out["torso"] = torso_angle(df, shoulder, hip)

    # interp_<métrica>: True si el frame usa algún keypoint interpolado de esa métrica
    for name, sides in TRIPLETS[EXERCISE].items():
        out[f"interp_{name}"] = filled[[f"x{i}" for i in sides[row.near_side]]].any(axis=1)
    out["interp_torso"] = filled[[f"x{shoulder}", f"x{hip}"]].any(axis=1)
    return out


def main():
    # keep_default_na=False: las celdas vacías se leen como "" en vez de NaN
    meta = pd.read_csv(METADATA, dtype=str, keep_default_na=False)
    todo = meta[(meta["subject_ids"] != "") & (meta["exercise"] == EXERCISE)]
    OUT_DIR.mkdir(exist_ok=True)

    summary = []
    for row in todo.itertuples():
        out = compute_metrics(row)
        stem = Path(row.file).stem
        out.to_csv(OUT_DIR / f"{stem}.csv", index=False)
        for metric in [c for c in out.columns if c != "frame" and not c.startswith("interp_")]:
            summary.append({
                "video": stem,
                "metrica": metric,
                "min": out[metric].min(),
                "max": out[metric].max(),
                "nan_%": 100 * out[metric].isna().mean(),
            })

    # Comprobación: rango de cada métrica y % de NaN que quedan
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
