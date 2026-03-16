from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
from openai import OpenAI

from app.config import OPENAI_API_KEY
from app.utils import leer_texto, dividir_en_fragmentos, listar_archivos_txt
from app.embeddings import (
    indexar_fragmentos,
    recuperar_fragmentos_semanticos,
    guardar_indice_vectorial,
    cargar_indice_vectorial,
    calcular_hash_texto,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def construir_ruta_indice(ruta_documento: str) -> str:
    """Construye una ruta de caché específica para cada documento."""
    nombre_base = Path(ruta_documento).stem
    return f"cache/{nombre_base}_indice_vectorial.json"


def preparar_indice_vectorial(
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
            return indice_vectorial, "cache"

    indice_vectorial = indexar_fragmentos(client, fragmentos)
    guardar_indice_vectorial(indice_vectorial, ruta_indice, hash_actual)
    return indice_vectorial, "regenerado"


def responder_pregunta(
    pregunta: str,
    indice_vectorial: list[tuple[int, str, list[float]]],
    top_k: int = 2
) -> tuple[str, list[tuple[int, str, float]]]:
    """Recupera fragmentos relevantes y genera respuesta."""
    resultados = recuperar_fragmentos_semanticos(
        client,
        pregunta,
        indice_vectorial,
        top_k=top_k
    )

    contexto_documental = "\n\n".join([fragmento for _, fragmento, _ in resultados])

    prompt = f"""
Responde a la pregunta del usuario usando únicamente la información del contexto documental.
Si el contexto documental no contiene la respuesta, di claramente que no aparece en el texto proporcionado.

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

    return response.output_text, resultados


def inicializar_estado() -> None:
    """Inicializa el estado de sesión necesario."""
    if "indice_vectorial" not in st.session_state:
        st.session_state["indice_vectorial"] = None

    if "origen_indice" not in st.session_state:
        st.session_state["origen_indice"] = None

    if "documento_activo" not in st.session_state:
        st.session_state["documento_activo"] = None

    if "historial_chat" not in st.session_state:
        st.session_state["historial_chat"] = []


def resetear_historial_si_cambia_documento(ruta_documento: str) -> None:
    """Limpia el historial visual si cambia el documento activo."""
    if st.session_state.get("documento_activo") != ruta_documento:
        st.session_state["historial_chat"] = []


def mostrar_historial_chat() -> None:
    """Muestra el historial visual de preguntas y respuestas."""
    historial = st.session_state.get("historial_chat", [])

    if not historial:
        st.info("Todavía no hay preguntas en esta sesión visual.")
        return

    for i, item in enumerate(historial, start=1):
        with st.container():
            st.markdown(f"### Pregunta {i}")
            st.markdown(f"**Usuario:** {item['pregunta']}")
            st.markdown("**Respuesta:**")
            st.write(item["respuesta"])

            with st.expander("Ver fragmentos recuperados"):
                for indice, fragmento, score in item["resultados"]:
                    st.markdown(f"**Fragmento {indice + 1} · score {score:.4f}**")
                    st.write(fragmento)
                    st.markdown("---")


def main() -> None:
    st.set_page_config(page_title="llm-text-lab", page_icon="🧠", layout="wide")
    inicializar_estado()

    st.title("🧠 llm-text-lab")
    st.caption("Asistente documental con recuperación semántica y caché por documento")

    try:
        archivos = listar_archivos_txt("data")
    except FileNotFoundError as e:
        st.error(str(e))
        return

    if not archivos:
        st.warning("No hay archivos .txt en la carpeta data/")
        return

    nombres_archivos = [archivo.name for archivo in archivos]
    archivo_seleccionado = st.sidebar.selectbox("Documento", nombres_archivos)

    ruta_documento = str(next(a for a in archivos if a.name == archivo_seleccionado))
    ruta_indice = construir_ruta_indice(ruta_documento)

    resetear_historial_si_cambia_documento(ruta_documento)

    texto = leer_texto(ruta_documento)
    fragmentos = dividir_en_fragmentos(texto, max_palabras=20)

    with st.sidebar:
        st.markdown("### Estado")
        st.write(f"**Documento:** {archivo_seleccionado}")
        st.write(f"**Fragmentos:** {len(fragmentos)}")

        if st.button("Preparar índice"):
            with st.spinner("Preparando índice vectorial..."):
                indice_vectorial, origen_indice = preparar_indice_vectorial(
                    texto, fragmentos, ruta_indice
                )
                st.session_state["indice_vectorial"] = indice_vectorial
                st.session_state["origen_indice"] = origen_indice
                st.session_state["documento_activo"] = ruta_documento

        if "origen_indice" in st.session_state and st.session_state["origen_indice"]:
            st.write(f"**Origen del índice:** {st.session_state['origen_indice']}")

        if st.button("Limpiar historial visual"):
            st.session_state["historial_chat"] = []
            st.success("Historial visual limpiado.")

    if (
        st.session_state.get("indice_vectorial") is None
        or st.session_state.get("documento_activo") != ruta_documento
    ):
        st.info("Pulsa **Preparar índice** en la barra lateral para comenzar.")
        return

    pregunta = st.text_input("Haz una pregunta sobre el documento")

    if st.button("Preguntar"):
        if not pregunta.strip():
            st.warning("Escribe una pregunta antes de continuar.")
        else:
            with st.spinner("Buscando fragmentos y generando respuesta..."):
                respuesta, resultados = responder_pregunta(
                    pregunta,
                    st.session_state["indice_vectorial"],
                    top_k=2
                )

            st.session_state["historial_chat"].append(
                {
                    "pregunta": pregunta,
                    "respuesta": respuesta,
                    "resultados": resultados,
                }
            )

    st.subheader("Conversación")
    mostrar_historial_chat()


if __name__ == "__main__":
    main()