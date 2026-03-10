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
from app.embeddings import similitud_coseno


MODOS_DISPONIBLES = [
    "resumen",
    "puntos_clave",
    "clasificacion",
    "tono",
    "todos",
    "pregunta_semantica",
]
MODOS_ANALISIS = ["resumen", "puntos_clave", "clasificacion", "tono"]


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


def obtener_embedding(client: OpenAI, texto: str) -> list[float]:
    """Obtiene el embedding de un texto usando la API."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding


def recuperar_fragmentos_semanticos(
    client: OpenAI,
    pregunta: str,
    fragmentos: list[str],
    top_k: int = 2
) -> list[tuple[int, str, float]]:
    """Recupera los fragmentos más similares semánticamente a una pregunta."""
    embedding_pregunta = obtener_embedding(client, pregunta)
    resultados = []

    for i, fragmento in enumerate(fragmentos):
        embedding_fragmento = obtener_embedding(client, fragmento)
        score = similitud_coseno(embedding_pregunta, embedding_fragmento)
        resultados.append((i, fragmento, score))

    resultados.sort(key=lambda x: x[2], reverse=True)
    return resultados[:top_k]


def ejecutar_pregunta_semantica(client: OpenAI, fragmentos: list[str]) -> None:
    """Responde una pregunta recuperando fragmentos por similitud semántica."""
    pregunta = input("\nEscribe tu pregunta sobre el texto: ").strip()

    resultados = recuperar_fragmentos_semanticos(client, pregunta, fragmentos, top_k=2)
    mostrar_resultados_semanticos(resultados)

    contexto = "\n\n".join([fragmento for _, fragmento, _ in resultados])
    prompt = construir_prompt_pregunta(contexto, pregunta)

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    mostrar_titulo("Respuesta a la pregunta")
    print(response.output_text)


def main() -> None:
    texto = leer_texto("data/ejemplo.txt")
    fragmentos = dividir_en_fragmentos(texto, max_palabras=20)

    mostrar_fragmentos(fragmentos)

    modo = pedir_modo()
    client = OpenAI(api_key=OPENAI_API_KEY)

    if modo == "pregunta_semantica":
        ejecutar_pregunta_semantica(client, fragmentos)
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