from openai import OpenAI
from app.utils import mostrar_titulo


def ejecutar_analisis(client: OpenAI, texto: str, modo: str) -> None:
    """Ejecuta un análisis y muestra el resultado."""
    if modo == "resumen":
        prompt = f"""
Analiza el siguiente texto y devuelve:

RESUMEN:
Un resumen breve y claro de 3 a 5 líneas.

Texto:
{texto}
"""
    elif modo == "puntos_clave":
        prompt = f"""
Analiza el siguiente texto y devuelve:

PUNTOS CLAVE:
- Punto 1
- Punto 2
- Punto 3

Texto:
{texto}
"""
    elif modo == "clasificacion":
        prompt = f"""
Analiza el siguiente texto y devuelve:

CLASIFICACIÓN TEMÁTICA:
Indica cuál es el tema principal del texto y explícalo brevemente.

Texto:
{texto}
"""
    elif modo == "tono":
        prompt = f"""
Analiza el siguiente texto y devuelve:

TONO DEL TEXTO:
Describe el tono del texto y explica brevemente por qué.

Texto:
{texto}
"""
    else:
        raise ValueError(f"Modo de análisis no válido: {modo}")

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    mostrar_titulo(f"Resultado del modo: {modo}")
    print(response.output_text)