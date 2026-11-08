"""
Calcula ángulos articulares frame a frame a partir del CSV de keypoints.
Ángulo interior en el vértice v: θ = arccos( (u−v)·(p−v) / (|u−v|·|p−v|) ), en 0-180°.
No depende de hacia dónde mire el sujeto (sustituye al ángulo orientado 0-360 del paper).
Salida: CSV con columnas frame, <nombre_angulo>...
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Configuración
# CSV_IN = Path("keypoints/subject_022_squat_good_side.csv")
CSV_IN = Path("keypoints/subject_022_squat_bad_side.csv")
EXERCISE = "squat"       # "squat" o "bicep_curl"
CONF_MIN = 0.5           # keypoints con menos confianza -> NaN
MAX_GAP = 5              # huecos de NaN de hasta MAX_GAP frames se interpolan
OUT_DIR = Path("angles")

# Tripletas (u, v, p) por ejercicio; v es el vértice del ángulo
TRIPLETS = {
    "squat": {"knee_left": (11, 13, 15), "knee_right": (12, 14, 16)},
    "bicep_curl": {"elbow_left": (5, 7, 9), "elbow_right": (6, 8, 10)},
}


def clean_keypoints(df):
    """Pone a NaN los keypoints poco fiables e interpola huecos cortos."""
    df = df.copy()
    for i in range(17):
        mask = df[f"c{i}"] < CONF_MIN
        df.loc[mask, [f"x{i}", f"y{i}"]] = np.nan
    df = df.interpolate(limit=MAX_GAP, limit_area="inside") 
    return df


def angle(df, u, v, p):
    """Ángulo interior en v (grados, 0-180) para todos los frames a la vez."""
    # Vectores desde el vértice v hacia u y hacia p, shape (frames, 2)
    a = df[[f"x{u}", f"y{u}"]].to_numpy() - df[[f"x{v}", f"y{v}"]].to_numpy()
    b = df[[f"x{p}", f"y{p}"]].to_numpy() - df[[f"x{v}", f"y{v}"]].to_numpy()
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    # clip: por redondeo cos puede salir 1.0000001 y arccos daría NaN
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    df = pd.read_csv(CSV_IN)
    df = clean_keypoints(df)

    out = pd.DataFrame({"frame": df["frame"]})
    for name, (u, v, p) in TRIPLETS[EXERCISE].items():
        out[name] = angle(df, u, v, p)
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / CSV_IN.name, index=False)
    # Comprobación: rango de cada ángulo y NaN que quedan
    print(out.describe().loc[["min", "max"]])
    print("NaN:", out.isna().sum().to_dict())


if __name__ == "__main__":
    main()
