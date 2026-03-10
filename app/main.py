from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.utils import leer_texto, mostrar_titulo
from app.prompts import construir_prompt


MODOS_DISPONIBLES = ["resumen", "puntos_clave", "clasificacion", "tono", "todos"]
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


def ejecutar_analisis(client: OpenAI, texto: str, modo: str) -> None:
    """Ejecuta un análisis y muestra el resultado."""
    prompt = construir_prompt(texto, modo)

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    mostrar_titulo(f"Resultado del modo: {modo}")
    print(response.output_text)


def main() -> None:
    texto = leer_texto("data/ejemplo.txt")
    modo = pedir_modo()

    client = OpenAI(api_key=OPENAI_API_KEY)

    if modo == "todos":
        for modo_individual in MODOS_ANALISIS:
            ejecutar_analisis(client, texto, modo_individual)
    else:
        ejecutar_analisis(client, texto, modo)


if __name__ == "__main__":
    main()