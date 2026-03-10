def construir_prompt(texto: str, modo: str) -> str:
    """Devuelve un prompt según el tipo de análisis solicitado."""

    prompts = {
        "resumen": f"""
Analiza el siguiente texto y devuelve:

RESUMEN:
Un resumen breve y claro de 3 a 5 líneas.

Texto:
{texto}
""",
        "puntos_clave": f"""
Analiza el siguiente texto y devuelve:

PUNTOS CLAVE:
- Punto 1
- Punto 2
- Punto 3

Texto:
{texto}
""",
        "clasificacion": f"""
Analiza el siguiente texto y devuelve:

CLASIFICACIÓN TEMÁTICA:
Indica cuál es el tema principal del texto y explícalo brevemente.

Texto:
{texto}
""",
        "tono": f"""
Analiza el siguiente texto y devuelve:

TONO DEL TEXTO:
Describe el tono del texto (por ejemplo: informativo, promocional, técnico, emocional, formal, etc.)
y explica brevemente por qué.

Texto:
{texto}
"""
    }

    if modo not in prompts:
        raise ValueError(
            f"Modo no válido: {modo}. Usa uno de estos: {', '.join(prompts.keys())}"
        )

    return prompts[modo]