from pathlib import Path
from app.utils import mostrar_titulo


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