from math import sqrt


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