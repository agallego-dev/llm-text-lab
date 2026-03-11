from pathlib import Path
from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.utils import (
    leer_texto,
    mostrar_titulo,
    dividir_en_fragmentos,
    mostrar_fragmentos,
    mostrar_resultados_semanticos,
    pedir_archivo_txt,
)
from app.prompts import construir_prompt, construir_prompt_pregunta
from app.embeddings import (
    indexar_fragmentos,
    recuperar_fragmentos_semanticos,
    guardar_indice_vectorial,
    cargar_indice_vectorial,
    calcular_hash_texto,
)


MODOS_DISPONIBLES = [
    "resumen",
    "puntos_clave",
    "clasificacion",
    "tono",
    "todos",
    "pregunta_semantica",
]
MODOS_ANALISIS = ["resumen", "puntos_clave", "clasificacion", "tono"]


def construir_ruta_indice(ruta_documento: str) -> str:
    """Construye una ruta de caché específica para cada documento."""
    nombre_base = Path(ruta_documento).stem
    return f"cache/{nombre_base}_indice_vectorial.json"


def pedir_modo() -> str:
    """Pide al usuario un modo de análisis válido, con reintento."""
    while True:
        print("Modos disponibles:")
        for modo in MODOS_DISPONIBLES:
            print(f"- {modo}")

        modo = input("\nElige un modo: ").strip().lower()

        if modo in MODOS_DISPONIBLES:
            return modo

        print(f"\nModo no válido. Debe ser uno de: {', '.join(MODOS_DISPONIBLES)}\n")


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


def ejecutar_analisis(client: OpenAI, texto: str, modo: str) -> None:
    """Ejecuta un análisis y muestra el resultado."""
    prompt = construir_prompt(texto, modo)

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    mostrar_titulo(f"Resultado del modo: {modo}")
    print(response.output_text)


def ejecutar_pregunta_semantica(
    client: OpenAI,
    indice_vectorial: list[tuple[int, str, list[float]]]
) -> None:
    """Permite hacer varias preguntas semánticas sobre el mismo texto."""
    mostrar_titulo("Modo pregunta semántica")
    print("Escribe tu pregunta sobre el documento.")
    print("Escribe 'salir' para terminar este modo.\n")

    while True:
        pregunta = input("Pregunta: ").strip()

        if not pregunta:
            print("No se ha escrito ninguna pregunta. Inténtalo de nuevo.\n")
            continue

        if pregunta.lower() == "salir":
            print("\nSaliendo del modo pregunta semántica.\n")
            break

        resultados = recuperar_fragmentos_semanticos(
            client,
            pregunta,
            indice_vectorial,
            top_k=2
        )
        mostrar_resultados_semanticos(resultados)

        contexto = "\n\n".join([fragmento for _, fragmento, _ in resultados])
        prompt = construir_prompt_pregunta(contexto, pregunta)

        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        mostrar_titulo("Respuesta a la pregunta")
        print(response.output_text)
        print()


def preparar_indice_vectorial(
    client: OpenAI,
    texto: str,
    fragmentos: list[str],
    ruta_indice: str
) -> list[tuple[int, str, list[float]]]:
    """Carga el índice desde caché si sigue siendo válido; si no, lo regenera."""
    hash_actual = calcular_hash_texto(texto)
    cache = cargar_indice_vectorial(ruta_indice)

    if cache is not None:
        hash_guardado, indice_vectorial = cache

        if hash_guardado == hash_actual:
            mostrar_titulo("Índice vectorial cargado desde caché")
            print(f"Se han cargado {len(indice_vectorial)} fragmentos desde JSON.\n")
            return indice_vectorial

        mostrar_titulo("Caché desactualizada")
        print("El documento ha cambiado. Se regenerará el índice.\n")

    mostrar_titulo("Indexando fragmentos")
    indice_vectorial = indexar_fragmentos(client, fragmentos)
    guardar_indice_vectorial(indice_vectorial, ruta_indice, hash_actual)
    print(f"Se han indexado y guardado {len(indice_vectorial)} fragmentos.\n")

    return indice_vectorial


def main() -> None:
    ruta_documento = pedir_archivo_txt("data")
    ruta_indice = construir_ruta_indice(ruta_documento)

    mostrar_titulo("Documento seleccionado")
    print(ruta_documento)

    texto = leer_texto(ruta_documento)
    fragmentos = dividir_en_fragmentos(texto, max_palabras=20)

    mostrar_fragmentos(fragmentos)

    client = OpenAI(api_key=OPENAI_API_KEY)
    indice_vectorial = preparar_indice_vectorial(client, texto, fragmentos, ruta_indice)

    modo = pedir_modo()

    if modo == "pregunta_semantica":
        ejecutar_pregunta_semantica(client, indice_vectorial)
        return

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