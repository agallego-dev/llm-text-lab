from openai import OpenAI
from app.utils import mostrar_titulo, mostrar_resultados_semanticos
from app.embeddings import recuperar_fragmentos_semanticos


def construir_contexto_conversacion(historial: list[dict], max_turnos: int = 3) -> str:
    """Construye un bloque de conversación reciente para incluirlo en el prompt."""
    if not historial:
        return ""

    ultimos_turnos = historial[-max_turnos:]
    lineas = ["CONVERSACIÓN RECIENTE:"]

    for item in ultimos_turnos:
        lineas.append(f"Usuario: {item['pregunta']}")
        lineas.append(f"Asistente: {item['respuesta']}")
        lineas.append("")

    return "\n".join(lineas)


def ejecutar_pregunta_semantica(
    client: OpenAI,
    indice_vectorial: list[tuple[int, str, list[float]]],
    historial: list[dict]
) -> None:
    """Permite hacer varias preguntas semánticas sobre el mismo texto con memoria reciente."""
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

        contexto_documental = "\n\n".join([fragmento for _, fragmento, _ in resultados])
        contexto_conversacion = construir_contexto_conversacion(historial, max_turnos=3)

        prompt = f"""
Responde a la pregunta del usuario usando únicamente la información del contexto documental.
Apóyate también en la conversación reciente si ayuda a entender referencias como "eso", "lo anterior", "y qué más", etc.
Si el contexto documental no contiene la respuesta, di claramente que no aparece en el texto proporcionado.

{contexto_conversacion}

CONTEXTO DOCUMENTAL:
{contexto_documental}

PREGUNTA DEL USUARIO:
{pregunta}

FORMATO DE RESPUESTA:
RESPUESTA:
...
"""

        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        respuesta = response.output_text

        historial.append(
            {
                "pregunta": pregunta,
                "resultados": resultados,
                "respuesta": respuesta,
            }
        )

        mostrar_titulo("Respuesta a la pregunta")
        print(respuesta)
        print()