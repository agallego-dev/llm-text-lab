import json
import hashlib
from math import sqrt
from pathlib import Path
from openai import OpenAI


def producto_punto(v1: list[float], v2: list[float]) -> float:
    """Calcula el producto punto entre dos vectores."""
    return sum(a * b for a, b in zip(v1, v2))


def norma(vector: list[float]) -> float:
    """Calcula la norma de un vector."""
    return sqrt(sum(x * x for x in vector))


def similitud_coseno(v1: list[float], v2: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    denominador = norma(v1) * norma(v2)

    if denominador == 0:
        return 0.0

    return producto_punto(v1, v2) / denominador


def obtener_embedding(client: OpenAI, texto: str) -> list[float]:
    """Obtiene el embedding de un texto usando la API."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texto
    )
    return response.data[0].embedding


def calcular_hash_texto(texto: str) -> str:
    """Calcula un hash SHA-256 del texto."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def indexar_fragmentos(
    client: OpenAI,
    fragmentos: list[str]
) -> list[tuple[int, str, list[float]]]:
    """
    Calcula y guarda los embeddings de los fragmentos una sola vez.
    Devuelve una lista de tuplas:
    (indice, fragmento, embedding)
    """
    indice_vectorial = []

    for i, fragmento in enumerate(fragmentos):
        embedding = obtener_embedding(client, fragmento)
        indice_vectorial.append((i, fragmento, embedding))

    return indice_vectorial


def guardar_indice_vectorial(
    indice_vectorial: list[tuple[int, str, list[float]]],
    ruta: str,
    hash_documento: str
) -> None:
    """Guarda el índice vectorial y el hash del documento en un archivo JSON."""
    path = Path(ruta)
    path.parent.mkdir(parents=True, exist_ok=True)

    datos = {
        "hash_documento": hash_documento,
        "fragmentos": [
            {
                "indice": indice,
                "fragmento": fragmento,
                "embedding": embedding
            }
            for indice, fragmento, embedding in indice_vectorial
        ]
    }

    path.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def cargar_indice_vectorial(
    ruta: str
) -> tuple[str, list[tuple[int, str, list[float]]]] | None:
    """Carga el hash y el índice vectorial desde un archivo JSON si existe."""
    path = Path(ruta)

    if not path.exists():
        return None

    datos = json.loads(path.read_text(encoding="utf-8"))

    hash_documento = datos["hash_documento"]
    fragmentos = [
        (item["indice"], item["fragmento"], item["embedding"])
        for item in datos["fragmentos"]
    ]

    return hash_documento, fragmentos


def recuperar_fragmentos_semanticos(
    client: OpenAI,
    pregunta: str,
    indice_vectorial: list[tuple[int, str, list[float]]],
    top_k: int = 2
) -> list[tuple[int, str, float]]:
    """
    Recupera los fragmentos más similares semánticamente a una pregunta,
    reutilizando embeddings ya calculados.
    """
    embedding_pregunta = obtener_embedding(client, pregunta)
    resultados = []

    for i, fragmento, embedding_fragmento in indice_vectorial:
        score = similitud_coseno(embedding_pregunta, embedding_fragmento)
        resultados.append((i, fragmento, score))

    resultados.sort(key=lambda x: x[2], reverse=True)
    return resultados[:top_k]