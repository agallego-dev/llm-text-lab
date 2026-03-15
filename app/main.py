from pathlib import Path
from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.utils import (
    leer_texto,
    mostrar_titulo,
    dividir_en_fragmentos,
    mostrar_fragmentos,
    pedir_archivo_txt,
)
from app.embeddings import (
    indexar_fragmentos,
    guardar_indice_vectorial,
    cargar_indice_vectorial,
    calcular_hash_texto,
)
from app.analysis import ejecutar_analisis
from app.chat import ejecutar_pregunta_semantica
from app.history import mostrar_historial, guardar_historial_en_txt


MODOS_ANALISIS = ["resumen", "puntos_clave", "clasificacion", "tono"]
MODOS_MENU = [
    "resumen",
    "puntos_clave",
    "clasificacion",
    "tono",
    "todos",
    "pregunta_semantica",
    "ver_historial",
    "guardar_historial",
    "ver_estado",
    "cambiar_documento",
    "salir",
]


def construir_ruta_indice(ruta_documento: str) -> str:
    """Construye una ruta de caché específica para cada documento."""
    nombre_base = Path(ruta_documento).stem
    return f"cache/{nombre_base}_indice_vectorial.json"


def pedir_modo_menu() -> str:
    """Pide un modo del menú principal del documento, con reintento."""
    while True:
        print("Opciones disponibles:")
        for modo in MODOS_MENU:
            print(f"- {modo}")

        modo = input("\nElige una opción: ").strip().lower()

        if modo in MODOS_MENU:
            return modo

        print(f"\nOpción no válida. Debe ser una de: {', '.join(MODOS_MENU)}\n")


def pedir_fragmento(num_fragmentos: int) -> int | None:
    """Permite elegir un fragmento concreto o todos, con reintento."""
    while True:
        seleccion = input(
            f"\nElige un fragmento (1-{num_fragmentos}) o escribe 'todos': "
        ).strip().lower()

        if seleccion == "todos":
            return None

        if not seleccion.isdigit():
            print("Debes escribir un número válido o 'todos'.")
            continue

        indice = int(seleccion)

        if indice < 1 or indice > num_fragmentos:
            print("El número de fragmento está fuera de rango.")
            continue

        return indice - 1


def preparar_indice_vectorial(
    client: OpenAI,
    texto: str,
    fragmentos: list[str],
    ruta_indice: str
) -> tuple[list[tuple[int, str, list[float]]], str]:
    """Carga el índice desde caché si sigue siendo válido; si no, lo regenera."""
    hash_actual = calcular_hash_texto(texto)
    cache = cargar_indice_vectorial(ruta_indice)

    if cache is not None:
        hash_guardado, indice_vectorial = cache

        if hash_guardado == hash_actual:
            mostrar_titulo("Índice vectorial cargado desde caché")
            print(f"Se han cargado {len(indice_vectorial)} fragmentos desde JSON.\n")
            return indice_vectorial, "cache"

        mostrar_titulo("Caché desactualizada")
        print("El documento ha cambiado. Se regenerará el índice.\n")

    mostrar_titulo("Indexando fragmentos")
    indice_vectorial = indexar_fragmentos(client, fragmentos)
    guardar_indice_vectorial(indice_vectorial, ruta_indice, hash_actual)
    print(f"Se han indexado y guardado {len(indice_vectorial)} fragmentos.\n")

    return indice_vectorial, "regenerado"


def mostrar_estado_sesion(
    ruta_documento: str,
    fragmentos: list[str],
    historial_documento: list[dict],
    ruta_indice: str,
    origen_indice: str
) -> None:
    """Muestra el estado actual del documento activo."""
    mostrar_titulo("Estado de la sesión")
    print(f"Documento activo: {ruta_documento}")
    print(f"Número de fragmentos: {len(fragmentos)}")
    print(f"Preguntas en historial del documento: {len(historial_documento)}")
    print(f"Ruta de caché: {ruta_indice}")
    print(f"Origen del índice actual: {origen_indice}\n")


def cargar_documento(
    client: OpenAI
) -> tuple[str, str, list[str], list[tuple[int, str, list[float]]], str, str]:
    """Carga un documento, sus fragmentos, su índice vectorial y metadatos."""
    ruta_documento = pedir_archivo_txt("data")
    ruta_indice = construir_ruta_indice(ruta_documento)

    mostrar_titulo("Documento seleccionado")
    print(ruta_documento)

    texto = leer_texto(ruta_documento)
    fragmentos = dividir_en_fragmentos(texto, max_palabras=20)

    mostrar_fragmentos(fragmentos)

    indice_vectorial, origen_indice = preparar_indice_vectorial(
        client, texto, fragmentos, ruta_indice
    )
    return ruta_documento, texto, fragmentos, indice_vectorial, ruta_indice, origen_indice


def main() -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)
    historiales_por_documento: dict[str, list[dict]] = {}

    documento_cargado = cargar_documento(client)
    (
        ruta_documento,
        texto,
        fragmentos,
        indice_vectorial,
        ruta_indice,
        origen_indice,
    ) = documento_cargado

    if ruta_documento not in historiales_por_documento:
        historiales_por_documento[ruta_documento] = []

    while True:
        historial_actual = historiales_por_documento[ruta_documento]
        modo = pedir_modo_menu()

        if modo == "salir":
            print("\nSaliendo del programa.")
            break

        if modo == "cambiar_documento":
            documento_cargado = cargar_documento(client)
            (
                ruta_documento,
                texto,
                fragmentos,
                indice_vectorial,
                ruta_indice,
                origen_indice,
            ) = documento_cargado

            if ruta_documento not in historiales_por_documento:
                historiales_por_documento[ruta_documento] = []

            continue

        if modo == "pregunta_semantica":
            ejecutar_pregunta_semantica(client, indice_vectorial, historial_actual)
            continue

        if modo == "ver_historial":
            mostrar_historial(historial_actual)
            continue

        if modo == "guardar_historial":
            nombre_base = Path(ruta_documento).stem
            ruta_salida = f"exports/{nombre_base}_historial.txt"
            guardar_historial_en_txt(historial_actual, ruta_salida)
            print(f"\nHistorial guardado en: {ruta_salida}\n")
            continue

        if modo == "ver_estado":
            mostrar_estado_sesion(
                ruta_documento,
                fragmentos,
                historial_actual,
                ruta_indice,
                origen_indice,
            )
            continue

        indice_fragmento = pedir_fragmento(len(fragmentos))

        if indice_fragmento is None:
            texto_objetivo = "\n\n".join(fragmentos)
        else:
            texto_objetivo = fragmentos[indice_fragmento]

        if modo == "todos":
            for modo_individual in MODOS_ANALISIS:
                ejecutar_analisis(client, texto_objetivo, modo_individual)
        else:
            ejecutar_analisis(client, texto_objetivo, modo)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError de archivo: {e}")
    except ValueError as e:
        print(f"\nError de validación: {e}")
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")