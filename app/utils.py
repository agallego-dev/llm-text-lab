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


def dividir_en_fragmentos(texto: str, max_palabras: int = 40) -> list[str]:
    """Divide un texto en fragmentos según un número máximo de palabras."""
    palabras = texto.split()
    fragmentos = []

    for i in range(0, len(palabras), max_palabras):
        fragmento = " ".join(palabras[i:i + max_palabras])
        fragmentos.append(fragmento)

    return fragmentos


def mostrar_fragmentos(fragmentos: list[str]) -> None:
    """Muestra por consola los fragmentos numerados."""
    mostrar_titulo("Fragmentos disponibles")

    for i, fragmento in enumerate(fragmentos, start=1):
        print(f"[{i}] {fragmento}\n")


def mostrar_resultados_semanticos(resultados: list[tuple[int, str, float]]) -> None:
    """Muestra por consola los fragmentos recuperados por similitud semántica."""
    mostrar_titulo("Fragmentos recuperados por similitud semántica")

    for indice, fragmento, score in resultados:
        print(f"[{indice + 1}] (score: {score:.4f}) {fragmento}\n")