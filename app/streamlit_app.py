from pathlib import Path
import sys
from datetime import datetime
import hashlib

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
from app.pdf_utils import extraer_texto_pdf_bytes

client = OpenAI(api_key=OPENAI_API_KEY)

MODOS_ANALISIS = ["resumen", "puntos_clave", "clasificacion", "tono"]


def calcular_hash_bytes(data: bytes) -> str:
    """Calcula hash SHA-256 de bytes."""
    return hashlib.sha256(data).hexdigest()


def construir_ruta_indice(clave_documento: str, max_palabras: int) -> str:
    """Construye una ruta de caché específica para un documento y chunk size."""
    nombre_seguro = clave_documento.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"cache/{nombre_seguro}_chunks_{max_palabras}_indice_vectorial.json"


def construir_ruta_export(clave_documento: str) -> str:
    """Construye una ruta de exportación específica para un documento."""
    nombre_seguro = clave_documento.replace("/", "_").replace("\\", "_").replace(":", "_")
    return f"exports/{nombre_seguro}_historial_streamlit.txt"


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
) -> tuple[str, list[tuple[int, str, float]], dict]:
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

    evaluacion = evaluar_recuperacion(resultados)
    return response.output_text, resultados, evaluacion


def evaluar_recuperacion(resultados: list[tuple[int, str, float]]) -> dict:
    """Calcula métricas simples de calidad de recuperación."""
    if not resultados:
        return {
            "num_fragmentos": 0,
            "score_medio": 0.0,
            "score_max": 0.0,
            "score_min": 0.0,
            "valoracion": "sin resultados",
        }

    scores = [score for _, _, score in resultados]
    score_medio = sum(scores) / len(scores)
    score_max = max(scores)
    score_min = min(scores)

    if score_medio >= 0.55:
        valoracion = "recuperación fuerte"
    elif score_medio >= 0.35:
        valoracion = "recuperación media"
    else:
        valoracion = "recuperación débil"

    return {
        "num_fragmentos": len(resultados),
        "score_medio": round(score_medio, 4),
        "score_max": round(score_max, 4),
        "score_min": round(score_min, 4),
        "valoracion": valoracion,
    }


def inicializar_estado() -> None:
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
    if "favoritos_por_documento" not in st.session_state:
        st.session_state["favoritos_por_documento"] = {}


def obtener_historial_documento(clave_documento: str) -> list[dict]:
    historiales = st.session_state["historiales_chat"]
    if clave_documento not in historiales:
        historiales[clave_documento] = []
    return historiales[clave_documento]


def obtener_favoritos_documento(clave_documento: str) -> list[str]:
    favoritos = st.session_state["favoritos_por_documento"]
    if clave_documento not in favoritos:
        favoritos[clave_documento] = []
    return favoritos[clave_documento]


def lanzar_pregunta_y_guardar(
    pregunta: str,
    indice_vectorial: list[tuple[int, str, list[float]]],
    historial_actual: list[dict],
    top_k: int,
    chunk_size: int,
) -> None:
    respuesta, resultados, evaluacion = responder_pregunta(
        pregunta,
        indice_vectorial,
        top_k=top_k,
    )

    historial_actual.append(
        {
            "pregunta": pregunta,
            "respuesta": respuesta,
            "resultados": resultados,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluacion": evaluacion,
        }
    )


def mostrar_bloque_evaluacion(evaluacion: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fragmentos", evaluacion["num_fragmentos"])
    col2.metric("Score medio", evaluacion["score_medio"])
    col3.metric("Score máx", evaluacion["score_max"])
    col4.metric("Score mín", evaluacion["score_min"])
    st.caption(f"Valoración: {evaluacion['valoracion']}")


def mostrar_historial_chat(historial: list[dict]) -> None:
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

            if "evaluacion" in item:
                with st.expander("Ver evaluación de recuperación"):
                    mostrar_bloque_evaluacion(item["evaluacion"])

            with st.expander("Ver fragmentos recuperados"):
                for indice, fragmento, score in item["resultados"]:
                    st.markdown(f"**Fragmento {indice + 1} · score {score:.4f}**")
                    st.write(fragmento)
                    st.markdown("---")


def mostrar_fragmentos_documento(fragmentos: list[str]) -> None:
    with st.expander("Ver fragmentos del documento"):
        for i, fragmento in enumerate(fragmentos, start=1):
            st.markdown(f"**Fragmento {i}**")
            st.write(fragmento)
            st.markdown("---")


def exportar_historial_a_txt(historial: list[dict], ruta_salida: str) -> None:
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

            if "evaluacion" in item:
                ev = item["evaluacion"]
                lineas.append(
                    f"Evaluación: {ev['valoracion']} | "
                    f"fragmentos={ev['num_fragmentos']} | "
                    f"score_medio={ev['score_medio']} | "
                    f"score_max={ev['score_max']} | "
                    f"score_min={ev['score_min']}\n"
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
    ruta_mostrada: str,
    texto: str,
    ruta_indice: str,
    fragmentos: list[str],
    indice_vectorial: list[tuple[int, str, list[float]]] | None,
    max_palabras: int,
) -> dict:
    path_idx = Path(ruta_indice)

    info = {
        "ruta_documento": ruta_mostrada,
        "hash_documento": calcular_hash_texto(texto),
        "fragmentos_documento": len(fragmentos),
        "chunk_size": max_palabras,
        "ruta_indice": ruta_indice,
        "indice_existe": path_idx.exists(),
        "tamano_indice_kb": None,
        "ultima_modificacion_indice": None,
        "vectores_indexados": len(indice_vectorial) if indice_vectorial else 0,
        "tamano_documento_kb": round(len(texto.encode("utf-8")) / 1024, 2),
    }

    if path_idx.exists():
        info["tamano_indice_kb"] = round(path_idx.stat().st_size / 1024, 2)
        info["ultima_modificacion_indice"] = datetime.fromtimestamp(
            path_idx.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

    return info


def mostrar_info_tecnica(info: dict) -> None:
    with st.expander("Información técnica del documento e índice"):
        st.markdown(f"**Ruta/origen del documento:** `{info['ruta_documento']}`")
        st.markdown(f"**Hash del documento:** `{info['hash_documento']}`")
        st.markdown(f"**Fragmentos del documento:** {info['fragmentos_documento']}")
        st.markdown(f"**Chunk size actual:** {info['chunk_size']} palabras")
        st.markdown(f"**Tamaño estimado del documento:** {info['tamano_documento_kb']} KB")
        st.markdown(f"**Ruta del índice:** `{info['ruta_indice']}`")
        st.markdown(f"**Índice existe:** {info['indice_existe']}")
        st.markdown(f"**Vectores indexados:** {info['vectores_indexados']}")
        st.markdown(f"**Tamaño del índice:** {info['tamano_indice_kb']} KB")
        st.markdown(f"**Última modificación del índice:** {info['ultima_modificacion_indice']}")


def obtener_resumen_rapido(clave_documento: str, texto: str) -> dict:
    cache = st.session_state["resumen_rapido_cache"]
    hash_doc = calcular_hash_texto(texto)
    clave = f"{clave_documento}:{hash_doc}"

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


def mostrar_resultados_comparacion(
    pregunta: str,
    indice_vectorial: list[tuple[int, str, list[float]]],
    top_k_a: int,
    top_k_b: int,
) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"## Comparación A · top_k = {top_k_a}")
        with st.spinner("Generando comparación A..."):
            respuesta_a, resultados_a, evaluacion_a = responder_pregunta(
                pregunta,
                indice_vectorial,
                top_k=top_k_a,
            )
        st.write(respuesta_a)
        mostrar_bloque_evaluacion(evaluacion_a)

        with st.expander("Ver fragmentos recuperados (A)"):
            for indice, fragmento, score in resultados_a:
                st.markdown(f"**Fragmento {indice + 1} · score {score:.4f}**")
                st.write(fragmento)
                st.markdown("---")

    with col2:
        st.markdown(f"## Comparación B · top_k = {top_k_b}")
        with st.spinner("Generando comparación B..."):
            respuesta_b, resultados_b, evaluacion_b = responder_pregunta(
                pregunta,
                indice_vectorial,
                top_k=top_k_b,
            )
        st.write(respuesta_b)
        mostrar_bloque_evaluacion(evaluacion_b)

        with st.expander("Ver fragmentos recuperados (B)"):
            for indice, fragmento, score in resultados_b:
                st.markdown(f"**Fragmento {indice + 1} · score {score:.4f}**")
                st.write(fragmento)
                st.markdown("---")


def obtener_documento_activo() -> tuple[str, str, str]:
    """Devuelve clave_documento, ruta/etiqueta mostrada y texto."""
    fuente = st.sidebar.radio(
        "Fuente del documento",
        ["TXT local", "PDF subido"],
        key="fuente_documento_radio"
    )

    if fuente == "TXT local":
        archivos = listar_archivos_txt("data")
        if not archivos:
            raise FileNotFoundError("No hay archivos .txt en la carpeta data/")

        nombres_archivos = [archivo.name for archivo in archivos]
        archivo_seleccionado = st.sidebar.selectbox("Documento TXT", nombres_archivos)

        ruta_documento = str(next(a for a in archivos if a.name == archivo_seleccionado))
        texto = leer_texto(ruta_documento)
        clave_documento = f"txt::{ruta_documento}"
        etiqueta = ruta_documento
        return clave_documento, etiqueta, texto

    uploaded_pdf = st.sidebar.file_uploader("Sube un PDF", type=["pdf"])

    if uploaded_pdf is None:
        return "", "PDF no cargado", ""

    pdf_bytes = uploaded_pdf.read()
    texto, paginas_con_error = extraer_texto_pdf_bytes(pdf_bytes)

    if paginas_con_error:
        st.sidebar.warning(
            f"No se pudo extraer texto de estas páginas: {', '.join(map(str, paginas_con_error))}"
        )

    hash_pdf = calcular_hash_bytes(pdf_bytes)
    clave_documento = f"pdf::{uploaded_pdf.name}::{hash_pdf}"
    etiqueta = f"PDF subido: {uploaded_pdf.name}"
    return clave_documento, etiqueta, texto


def main() -> None:
    st.set_page_config(page_title="llm-text-lab", page_icon="🧠", layout="wide")
    inicializar_estado()

    st.title("🧠 llm-text-lab")
    st.caption("Asistente documental con recuperación semántica y caché por documento")

    try:
        clave_documento, etiqueta_documento, texto = obtener_documento_activo()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Error al cargar el documento: {e}")
        return

    if not clave_documento:
        st.info("Selecciona un documento TXT o sube un PDF para empezar.")
        return

    if not texto.strip():
        st.warning("No se ha podido extraer texto útil del documento seleccionado.")
        return

    ruta_export = construir_ruta_export(clave_documento)

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

    ruta_indice = construir_ruta_indice(clave_documento, max_palabras)
    fragmentos = dividir_en_fragmentos(texto, max_palabras=max_palabras)
    historial_actual = obtener_historial_documento(clave_documento)
    favoritos_actuales = obtener_favoritos_documento(clave_documento)

    with st.sidebar:
        st.markdown("### Estado")
        st.write(f"**Documento:** {etiqueta_documento}")
        st.write(f"**Fragmentos:** {len(fragmentos)}")
        st.write(f"**Preguntas del documento:** {len(historial_actual)}")
        st.write(f"**Favoritos del documento:** {len(favoritos_actuales)}")

        if st.button("Preparar índice"):
            with st.spinner("Preparando índice vectorial..."):
                indice_vectorial, origen_indice = preparar_indice_vectorial(
                    texto, fragmentos, ruta_indice
                )
                st.session_state["indice_vectorial"] = indice_vectorial
                st.session_state["origen_indice"] = origen_indice
                st.session_state["documento_activo"] = clave_documento
                st.session_state["chunk_size_activo"] = max_palabras

        if st.button("Reindexar manualmente"):
            with st.spinner("Regenerando índice vectorial..."):
                indice_vectorial, origen_indice = preparar_indice_vectorial(
                    texto, fragmentos, ruta_indice, forzar_reindexado=True
                )
                st.session_state["indice_vectorial"] = indice_vectorial
                st.session_state["origen_indice"] = origen_indice
                st.session_state["documento_activo"] = clave_documento
                st.session_state["chunk_size_activo"] = max_palabras
                st.success("Índice regenerado correctamente.")

        if st.button("Limpiar historial visual de este documento"):
            st.session_state["historiales_chat"][clave_documento] = []
            st.success("Historial visual del documento limpiado.")

        if st.button("Exportar historial de este documento"):
            exportar_historial_a_txt(historial_actual, ruta_export)
            st.success(f"Historial exportado en: {ruta_export}")

        if st.session_state.get("origen_indice"):
            st.write(f"**Origen del índice:** {st.session_state['origen_indice']}")

    mostrar_fragmentos_documento(fragmentos)

    if (
        st.session_state.get("indice_vectorial") is None
        or st.session_state.get("documento_activo") != clave_documento
        or st.session_state.get("chunk_size_activo") != max_palabras
    ):
        st.info("Pulsa **Preparar índice** en la barra lateral para comenzar con la configuración actual.")
        return

    info_indice = obtener_info_indice(
        etiqueta_documento,
        texto,
        ruta_indice,
        fragmentos,
        st.session_state.get("indice_vectorial"),
        max_palabras,
    )
    mostrar_info_tecnica(info_indice)

    tab_chat, tab_analisis, tab_resumen, tab_comparador, tab_favoritos = st.tabs(
        ["Pregunta semántica", "Análisis del documento", "Resumen rápido", "Comparador", "Favoritos"]
    )

    with tab_chat:
        pregunta = st.text_input("Haz una pregunta sobre el documento", key="pregunta_semantica_input")

        col_preguntar, col_favorito = st.columns([2, 1])

        with col_preguntar:
            preguntar_click = st.button("Preguntar")

        with col_favorito:
            guardar_favorito_click = st.button("Guardar como favorita")

        if preguntar_click:
            if not pregunta.strip():
                st.warning("Escribe una pregunta antes de continuar.")
            else:
                with st.spinner("Buscando fragmentos y generando respuesta..."):
                    lanzar_pregunta_y_guardar(
                        pregunta,
                        st.session_state["indice_vectorial"],
                        historial_actual,
                        top_k,
                        max_palabras,
                    )

        if guardar_favorito_click:
            if not pregunta.strip():
                st.warning("Escribe una pregunta antes de guardarla.")
            elif pregunta in favoritos_actuales:
                st.info("Esa pregunta ya está guardada como favorita.")
            else:
                favoritos_actuales.append(pregunta)
                st.success("Pregunta guardada como favorita.")

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
                resumen_rapido = obtener_resumen_rapido(clave_documento, texto)

            st.markdown("### Resumen")
            st.write(resumen_rapido["resumen"])

            st.markdown("### Clasificación")
            st.write(resumen_rapido["clasificacion"])

            st.markdown("### Tono")
            st.write(resumen_rapido["tono"])

            col1, col2 = st.columns(2)
            col1.metric("Fragmentos actuales", len(fragmentos))
            col2.metric("Chunk size actual", max_palabras)

    with tab_comparador:
        st.subheader("Comparador de configuraciones")

        pregunta_comp = st.text_input(
            "Escribe una pregunta para comparar respuestas",
            key="pregunta_comparador_input"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            top_k_a = st.slider("top_k A", min_value=1, max_value=5, value=1, key="topk_a")
        with col_b:
            top_k_b = st.slider("top_k B", min_value=1, max_value=5, value=3, key="topk_b")

        if st.button("Comparar respuestas"):
            if not pregunta_comp.strip():
                st.warning("Escribe una pregunta para comparar.")
            elif top_k_a == top_k_b:
                st.warning("Elige valores distintos de top_k para que la comparación tenga sentido.")
            else:
                mostrar_resultados_comparacion(
                    pregunta_comp,
                    st.session_state["indice_vectorial"],
                    top_k_a,
                    top_k_b,
                )

    with tab_favoritos:
        st.subheader("Preguntas favoritas del documento")

        if not favoritos_actuales:
            st.info("Todavía no hay preguntas favoritas para este documento.")
        else:
            for i, favorita in enumerate(favoritos_actuales, start=1):
                col_texto, col_lanzar = st.columns([4, 1])
                with col_texto:
                    st.markdown(f"**{i}.** {favorita}")
                with col_lanzar:
                    if st.button("Lanzar", key=f"fav_{clave_documento}_{i}"):
                        with st.spinner("Ejecutando pregunta favorita..."):
                            lanzar_pregunta_y_guardar(
                                favorita,
                                st.session_state["indice_vectorial"],
                                historial_actual,
                                top_k,
                                max_palabras,
                            )
                        st.success("Pregunta favorita ejecutada y añadida al historial.")


if __name__ == "__main__":
    main()