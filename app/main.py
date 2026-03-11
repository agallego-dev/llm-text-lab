from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.utils import (
    leer_texto,
    mostrar_titulo,
    dividir_en_fragmentos,
    mostrar_fragmentos,
    mostrar_resultados_semanticos,
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
RUTA_INDICE = "cache/indice_vectorial.json"


def pedir_modo() -> str:
    """Pide al usuario un modo de análisis válido."""
    print("Modos disponibles:")
    for modo in MODOS_DISPONIBLES:
        print(f"- {modo}")

    modo = input("\nElige un modo: ").strip().lower()

    if modo not in MODOS_DISPONIBLES:
        raise ValueError(
            f"Modo no válido. Debe ser uno de: {', '.join(MODOS_DISPONIBLES)}"
        )

    return modo


def pedir_fragmento(num_fragmentos: int) -> int | None:
    """Permite elegir un fragmento concreto o todos."""
    seleccion = input(
        f"\nElige un fragmento (1-{num_fragmentos}) o escribe 'todos': "
    ).strip().lower()

    if seleccion == "todos":
        return None

    if not seleccion.isdigit():
        raise ValueError("Debes escribir un número válido o 'todos'.")

    indice = int(seleccion)

    if indice < 1 or indice > num_fragmentos:
        raise ValueError("El número de fragmento está fuera de rango.")

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
    """Responde una pregunta recuperando fragmentos por similitud semántica."""
    pregunta = input("\nEscribe tu pregunta sobre el texto: ").strip()

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


def preparar_indice_vectorial(
    client: OpenAI,
    texto: str,
    fragmentos: list[str]
) -> list[tuple[int, str, list[float]]]:
    """Carga el índice desde caché si sigue siendo válido; si no, lo regenera."""
    hash_actual = calcular_hash_texto(texto)
    cache = cargar_indice_vectorial(RUTA_INDICE)

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
    guardar_indice_vectorial(indice_vectorial, RUTA_INDICE, hash_actual)
    print(f"Se han indexado y guardado {len(indice_vectorial)} fragmentos.\n")

    return indice_vectorial


def main() -> None:
    texto = leer_texto("data/ejemplo.txt")
    fragmentos = dividir_en_fragmentos(texto, max_palabras=20)

    mostrar_fragmentos(fragmentos)

    client = OpenAI(api_key=OPENAI_API_KEY)
    indice_vectorial = preparar_indice_vectorial(client, texto, fragmentos)

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
    main()