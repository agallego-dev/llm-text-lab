from openai import OpenAI
from app.utils import mostrar_titulo


def construir_prompt_analisis(texto: str, modo: str) -> str:
    """Construye el prompt de análisis según el modo."""
    if modo == "resumen":
        return f"""
Analiza el siguiente texto y devuelve:

RESUMEN:
Un resumen breve y claro de 3 a 5 líneas.

Texto:
{texto}
"""
    elif modo == "puntos_clave":
        return f"""
Analiza el siguiente texto y devuelve:

PUNTOS CLAVE:
- Punto 1
- Punto 2
- Punto 3

Texto:
{texto}
"""
    elif modo == "clasificacion":
        return f"""
Analiza el siguiente texto y devuelve:

CLASIFICACIÓN TEMÁTICA:
Indica cuál es el tema principal del texto y explícalo brevemente.

Texto:
{texto}
"""
    elif modo == "tono":
        return f"""
Analiza el siguiente texto y devuelve:

TONO DEL TEXTO:
Describe el tono del texto y explica brevemente por qué.

Texto:
{texto}
"""
    else:
        raise ValueError(f"Modo de análisis no válido: {modo}")


def obtener_analisis(client: OpenAI, texto: str, modo: str) -> str:
    """Devuelve el resultado del análisis como texto."""
    prompt = construir_prompt_analisis(texto, modo)

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text


def ejecutar_analisis(client: OpenAI, texto: str, modo: str) -> None:
    """Ejecuta un análisis y lo muestra por consola."""
    resultado = obtener_analisis(client, texto, modo)

    mostrar_titulo(f"Resultado del modo: {modo}")
    print(resultado)