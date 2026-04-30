from pathlib import Path
import sys
from datetime import datetime

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
from app.analysis import obtener_analisis

client = OpenAI(api_key=OPENAI_API_KEY)

MODOS_ANALISIS = ["resumen", "puntos_clave", "clasificacion", "tono"]


def construir_ruta_indice(ruta_documento: str, max_palabras: int) -> str:
    """Construye una ruta de caché específica para cada documento y chunk size."""
    nombre_base = Path(ruta_documento).stem
    return f"cache/{nombre_base}_chunks_{max_palabras}_indice_vectorial.json"


def construir_ruta_export(ruta_documento: str) -> str:
    """Construye una ruta de exportación específica para cada documento."""
    nombre_base = Path(ruta_documento).stem
    return f"exports/{nombre_base}_historial_streamlit.txt"


def preparar_indice_vectorial(
    texto: str,
    fragmentos: list[str],
    ruta_indice: str,
    forzar_reindexado: bool = False,
) -> tuple[list[tuple[int, str, list[float]]], str]:
    """Carga el índice desde caché si sigue siendo válido; si no, lo regenera."""
    hash_actual = calcular_hash_texto(texto)

    if not forzar_reindexado:
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
    top_k: int = 2,
) -> tuple[str, list[tuple[int, str, float]]]:
    """Recupera fragmentos relevantes y genera respuesta."""
    resultados = recuperar_fragmentos_semanticos(
        client,
        pregunta,
        indice_vectorial,
        top_k=top_k,
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

    if "historiales_chat" not in st.session_state:
        st.session_state["historiales_chat"] = {}

    if "chunk_size_activo" not in st.session_state:
        st.session_state["chunk_size_activo"] = None

    if "resumen_rapido_cache" not in st.session_state:
        st.session_state["resumen_rapido_cache"] = {}


def obtener_historial_documento(ruta_documento: str) -> list[dict]:
    """Obtiene o crea el historial visual del documento actual."""
    historiales = st.session_state["historiales_chat"]

    if ruta_documento not in historiales:
        historiales[ruta_documento] = []

    return historiales[ruta_documento]


def mostrar_historial_chat(historial: list[dict]) -> None:
    """Muestra el historial visual de preguntas y respuestas."""
    if not historial:
        st.info("Todavía no hay preguntas en la sesión de este documento.")
        return

    for i, item in enumerate(historial, start=1):
        with st.container():
            st.markdown(f"### Pregunta {i}")
            st.markdown(f"**Usuario:** {item['pregunta']}")
            st.caption(
                f"Hora: {item['timestamp']} · chunk_size: {item['chunk_size']} · top_k: {item['top_k']}"
            )
            st.markdown("**Respuesta:**")
            st.write(item["respuesta"])

            with st.expander("Ver fragmentos recuperados"):
                for indice, fragmento, score in item["resultados"]:
                    st.markdown(f"**Fragmento {indice + 1} · score {score:.4f}**")
                    st.write(fragmento)
                    st.markdown("---")


def mostrar_fragmentos_documento(fragmentos: list[str]) -> None:
    """Muestra todos los fragmentos del documento."""
    with st.expander("Ver fragmentos del documento"):
        for i, fragmento in enumerate(fragmentos, start=1):
            st.markdown(f"**Fragmento {i}**")
            st.write(fragmento)
            st.markdown("---")


def exportar_historial_a_txt(historial: list[dict], ruta_salida: str) -> None:
    """Exporta el historial del documento actual a un fichero de texto."""
    path = Path(ruta_salida)
    path.parent.mkdir(parents=True, exist_ok=True)

    lineas = []

    if not historial:
        lineas.append("No hay preguntas registradas para este documento.\n")
    else:
        lineas.append("HISTORIAL STREAMLIT DEL DOCUMENTO\n")
        lineas.append("=" * 50 + "\n")

        for i, item in enumerate(historial, start=1):
            lineas.append(f"[{i}] Pregunta: {item['pregunta']}\n")
            lineas.append(
                f"Hora: {item['timestamp']} | chunk_size: {item['chunk_size']} | top_k: {item['top_k']}\n"
            )
            lineas.append("Fragmentos recuperados:\n")

            for indice, fragmento, score in item["resultados"]:
                lineas.append(
                    f"  - [{indice + 1}] (score: {score:.4f}) {fragmento}\n"
                )

            lineas.append("Respuesta:\n")
            lineas.append(f"{item['respuesta']}\n")
            lineas.append("-" * 50 + "\n")

    path.write_text("".join(lineas), encoding="utf-8")


def obtener_info_indice(
    ruta_documento: str,
    ruta_indice: str,
    fragmentos: list[str],
    indice_vectorial: list[tuple[int, str, list[float]]] | None,
    max_palabras: int,
) -> dict:
    """Devuelve metadatos técnicos del documento e índice."""
    path_doc = Path(ruta_documento)
    path_idx = Path(ruta_indice)

    info = {
        "ruta_documento": ruta_documento,
        "hash_documento": calcular_hash_texto(leer_texto(ruta_documento)),
        "fragmentos_documento": len(fragmentos),
        "chunk_size": max_palabras,
        "ruta_indice": ruta_indice,
        "indice_existe": path_idx.exists(),
        "tamano_indice_kb": None,
        "ultima_modificacion_indice": None,
        "vectores_indexados": len(indice_vectorial) if indice_vectorial else 0,
        "tamano_documento_kb": round(path_doc.stat().st_size / 1024, 2) if path_doc.exists() else None,
    }

    if path_idx.exists():
        info["tamano_indice_kb"] = round(path_idx.stat().st_size / 1024, 2)
        info["ultima_modificacion_indice"] = datetime.fromtimestamp(
            path_idx.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

    return info


def mostrar_info_tecnica(info: dict) -> None:
    """Muestra información técnica del documento e índice."""
    with st.expander("Información técnica del documento e índice"):
        st.markdown(f"**Ruta del documento:** `{info['ruta_documento']}`")
        st.markdown(f"**Hash del documento:** `{info['hash_documento']}`")
        st.markdown(f"**Fragmentos del documento:** {info['fragmentos_documento']}")
        st.markdown(f"**Chunk size actual:** {info['chunk_size']} palabras")
        st.markdown(f"**Tamaño del documento:** {info['tamano_documento_kb']} KB")
        st.markdown(f"**Ruta del índice:** `{info['ruta_indice']}`")
        st.markdown(f"**Índice existe:** {info['indice_existe']}")
        st.markdown(f"**Vectores indexados:** {info['vectores_indexados']}")
        st.markdown(f"**Tamaño del índice:** {info['tamano_indice_kb']} KB")
        st.markdown(f"**Última modificación del índice:** {info['ultima_modificacion_indice']}")


def obtener_resumen_rapido(ruta_documento: str, texto: str) -> dict:
    """Genera o recupera un resumen rápido del documento actual."""
    cache = st.session_state["resumen_rapido_cache"]
    hash_doc = calcular_hash_texto(texto)
    clave = f"{ruta_documento}:{hash_doc}"

    if clave in cache:
        return cache[clave]

    resumen = obtener_analisis(client, texto, "resumen")
    clasificacion = obtener_analisis(client, texto, "clasificacion")
    tono = obtener_analisis(client, texto, "tono")

    resultado = {
        "resumen": resumen,
        "clasificacion": clasificacion,
        "tono": tono,
    }
    cache[clave] = resultado
    return resultado


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
    ruta_export = construir_ruta_export(ruta_documento)

    with st.sidebar:
        st.markdown("### Configuración de recuperación")

        max_palabras = st.slider(
            "Tamaño de fragmento (palabras)",
            min_value=10,
            max_value=100,
            value=20,
            step=5,
        )

        top_k = st.slider(
            "Número de fragmentos a recuperar",
            min_value=1,
            max_value=5,
            value=2
        )

    ruta_indice = construir_ruta_indice(ruta_documento, max_palabras)
    texto = leer_texto(ruta_documento)
    fragmentos = dividir_en_fragmentos(texto, max_palabras=max_palabras)
    historial_actual = obtener_historial_documento(ruta_documento)

    with st.sidebar:
        st.markdown("### Estado")
        st.write(f"**Documento:** {archivo_seleccionado}")
        st.write(f"**Fragmentos:** {len(fragmentos)}")
        st.write(f"**Preguntas del documento:** {len(historial_actual)}")

        if st.button("Preparar índice"):
            with st.spinner("Preparando índice vectorial..."):
                indice_vectorial, origen_indice = preparar_indice_vectorial(
                    texto, fragmentos, ruta_indice
                )
                st.session_state["indice_vectorial"] = indice_vectorial
                st.session_state["origen_indice"] = origen_indice
                st.session_state["documento_activo"] = ruta_documento
                st.session_state["chunk_size_activo"] = max_palabras

        if st.button("Reindexar manualmente"):
            with st.spinner("Regenerando índice vectorial..."):
                indice_vectorial, origen_indice = preparar_indice_vectorial(
                    texto, fragmentos, ruta_indice, forzar_reindexado=True
                )
                st.session_state["indice_vectorial"] = indice_vectorial
                st.session_state["origen_indice"] = origen_indice
                st.session_state["documento_activo"] = ruta_documento
                st.session_state["chunk_size_activo"] = max_palabras
                st.success("Índice regenerado correctamente.")

        if st.button("Limpiar historial visual de este documento"):
            st.session_state["historiales_chat"][ruta_documento] = []
            st.success("Historial visual del documento limpiado.")

        if st.button("Exportar historial de este documento"):
            exportar_historial_a_txt(historial_actual, ruta_export)
            st.success(f"Historial exportado en: {ruta_export}")

        if st.session_state.get("origen_indice"):
            st.write(f"**Origen del índice:** {st.session_state['origen_indice']}")

    mostrar_fragmentos_documento(fragmentos)

    if (
        st.session_state.get("indice_vectorial") is None
        or st.session_state.get("documento_activo") != ruta_documento
        or st.session_state.get("chunk_size_activo") != max_palabras
    ):
        st.info("Pulsa **Preparar índice** en la barra lateral para comenzar con la configuración actual.")
        return

    info_indice = obtener_info_indice(
        ruta_documento,
        ruta_indice,
        fragmentos,
        st.session_state.get("indice_vectorial"),
        max_palabras,
    )
    mostrar_info_tecnica(info_indice)

    tab_chat, tab_analisis, tab_resumen = st.tabs(
        ["Pregunta semántica", "Análisis del documento", "Resumen rápido"]
    )

    with tab_chat:
        pregunta = st.text_input("Haz una pregunta sobre el documento", key="pregunta_semantica_input")

        if st.button("Preguntar"):
            if not pregunta.strip():
                st.warning("Escribe una pregunta antes de continuar.")
            else:
                with st.spinner("Buscando fragmentos y generando respuesta..."):
                    respuesta, resultados = responder_pregunta(
                        pregunta,
                        st.session_state["indice_vectorial"],
                        top_k=top_k,
                    )

                historial_actual.append(
                    {
                        "pregunta": pregunta,
                        "respuesta": respuesta,
                        "resultados": resultados,
                        "top_k": top_k,
                        "chunk_size": max_palabras,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        st.subheader("Conversación del documento actual")
        mostrar_historial_chat(historial_actual)

    with tab_analisis:
        st.subheader("Análisis directo del documento")

        opciones_fragmento = ["Todo el documento"] + [
            f"Fragmento {i + 1}" for i in range(len(fragmentos))
        ]

        fragmento_seleccionado = st.selectbox(
            "Selecciona qué parte analizar",
            opciones_fragmento,
            key="fragmento_analisis_select"
        )

        modo_analisis = st.selectbox(
            "Modo de análisis",
            MODOS_ANALISIS + ["todos"],
            key="modo_analisis_select"
        )

        if st.button("Analizar"):
            if fragmento_seleccionado == "Todo el documento":
                texto_objetivo = "\n\n".join(fragmentos)
            else:
                indice = int(fragmento_seleccionado.split()[-1]) - 1
                texto_objetivo = fragmentos[indice]

            with st.spinner("Generando análisis..."):
                if modo_analisis == "todos":
                    resultados_analisis = {}
                    for modo in MODOS_ANALISIS:
                        resultados_analisis[modo] = obtener_analisis(client, texto_objetivo, modo)

                    st.subheader("Resultados del análisis")
                    for modo, resultado in resultados_analisis.items():
                        with st.expander(f"Modo: {modo}"):
                            st.write(resultado)
                else:
                    resultado = obtener_analisis(client, texto_objetivo, modo_analisis)
                    st.subheader(f"Resultado del modo: {modo_analisis}")
                    st.write(resultado)

    with tab_resumen:
        st.subheader("Vista general del documento")

        if st.button("Generar resumen rápido"):
            with st.spinner("Generando resumen general del documento..."):
                resumen_rapido = obtener_resumen_rapido(ruta_documento, texto)

            st.markdown("### Resumen")
            st.write(resumen_rapido["resumen"])

            st.markdown("### Clasificación")
            st.write(resumen_rapido["clasificacion"])

            st.markdown("### Tono")
            st.write(resumen_rapido["tono"])

            col1, col2 = st.columns(2)
            col1.metric("Fragmentos actuales", len(fragmentos))
            col2.metric("Chunk size actual", max_palabras)


if __name__ == "__main__":
    main()