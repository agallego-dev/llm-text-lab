from pathlib import Path

def leer_texto(ruta: str) -> str:
    path = Path(ruta)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    return path.read_text(encoding="utf-8")