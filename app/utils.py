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


def listar_archivos_txt(directorio: str) -> list[Path]:
    """Lista los archivos .txt de un directorio."""
    path = Path(directorio)

    if not path.exists():
        raise FileNotFoundError(f"No existe el directorio: {directorio}")

    return sorted(path.glob("*.txt"))


def pedir_archivo_txt(directorio: str = "data") -> str:
    """Muestra los archivos .txt disponibles y permite elegir uno, con reintento."""
    archivos = listar_archivos_txt(directorio)

    if not archivos:
        raise FileNotFoundError(f"No hay archivos .txt en el directorio: {directorio}")

    while True:
        mostrar_titulo("Archivos disponibles")

        for i, archivo in enumerate(archivos, start=1):
            print(f"[{i}] {archivo.name}")

        seleccion = input("\nElige un archivo por número: ").strip()

        if not seleccion.isdigit():
            print("Entrada no válida. Debes introducir un número.\n")
            continue

        indice = int(seleccion)

        if indice < 1 or indice > len(archivos):
            print("Número fuera de rango. Inténtalo de nuevo.\n")
            continue

        return str(archivos[indice - 1])