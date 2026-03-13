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
from app.embeddings import (
    indexar_fragmentos,
    recuperar_fragmentos_semanticos,
    guardar_indice_vectorial,
    cargar_indice_vectorial,
    calcular_hash_texto,
)


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


def mostrar_historial(historial: list[dict]) -> None:
    """Muestra el historial de preguntas y respuestas del documento actual."""
    mostrar_titulo("Historial del documento actual")

    if not historial:
        print("Todavía no hay preguntas registradas para este documento.\n")
        return

    for i, item in enumerate(historial, start=1):
        print(f"[{i}] Pregunta: {item['pregunta']}")
        print("Fragmentos recuperados:")

        for indice, fragmento, score in item["resultados"]:
            print(f"  - [{indice + 1}] (score: {score:.4f}) {fragmento}")

        print("Respuesta:")
        print(item["respuesta"])
        print("-" * 50)


def guardar_historial_en_txt(historial: list[dict], ruta_salida: str) -> None:
    """Guarda el historial del documento actual en un archivo de texto."""
    path = Path(ruta_salida)
    path.parent.mkdir(parents=True, exist_ok=True)

    lineas = []

    if not historial:
        lineas.append("No hay preguntas registradas para este documento.\n")
    else:
        lineas.append("HISTORIAL DEL DOCUMENTO ACTUAL\n")
        lineas.append("=" * 50 + "\n")

        for i, item in enumerate(historial, start=1):
            lineas.append(f"[{i}] Pregunta: {item['pregunta']}\n")
            lineas.append("Fragmentos recuperados:\n")

            for indice, fragmento, score in item["resultados"]:
                lineas.append(
                    f"  - [{indice + 1}] (score: {score:.4f}) {fragmento}\n"
                )

            lineas.append("Respuesta:\n")
            lineas.append(f"{item['respuesta']}\n")
            lineas.append("-" * 50 + "\n")

    path.write_text("".join(lineas), encoding="utf-8")


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