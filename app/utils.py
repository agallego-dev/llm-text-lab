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


def normalizar_texto(texto: str) -> list[str]:
    """Pasa el texto a minúsculas y lo separa en palabras simples."""
    texto = texto.lower()
    for simbolo in ",.;:!?()[]{}\"'":
        texto = texto.replace(simbolo, "")
    return texto.split()


def buscar_fragmentos_relevantes(
    pregunta: str,
    fragmentos: list[str],
    top_k: int = 2
) -> list[tuple[int, str, int]]:
    """
    Busca los fragmentos más relevantes según coincidencia de palabras.
    Devuelve una lista de tuplas: (indice, fragmento, puntuacion).
    """
    palabras_pregunta = set(normalizar_texto(pregunta))
    resultados = []

    for i, fragmento in enumerate(fragmentos):
        palabras_fragmento = set(normalizar_texto(fragmento))
        puntuacion = len(palabras_pregunta.intersection(palabras_fragmento))
        resultados.append((i, fragmento, puntuacion))

    resultados.sort(key=lambda x: x[2], reverse=True)

    return [r for r in resultados if r[2] > 0][:top_k]


def mostrar_resultados_busqueda(resultados: list[tuple[int, str, int]]) -> None:
    """Muestra los fragmentos recuperados con su puntuación."""
    mostrar_titulo("Fragmentos recuperados")

    if not resultados:
        print("No se encontraron fragmentos claramente relacionados con la pregunta.\n")
        return

    for indice, fragmento, puntuacion in resultados:
        print(f"[{indice + 1}] (coincidencias: {puntuacion}) {fragmento}\n")