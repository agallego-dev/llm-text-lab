from pathlib import Path


def leer_texto(ruta: str) -> str:
    """Lee un fichero de texto y devuelve su contenido."""
    path = Path(ruta)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    return path.read_text(encoding="utf-8")


def mostrar_titulo(titulo: str) -> None:
    """Muestra un título formateado por consola."""
    print("\n" + "=" * 50)
    print(titulo.upper())
    print("=" * 50)