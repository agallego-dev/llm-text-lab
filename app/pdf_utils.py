from io import BytesIO
from pypdf import PdfReader


def extraer_texto_pdf_bytes(pdf_bytes: bytes) -> tuple[str, list[int]]:
    """
    Extrae texto de un PDF a partir de bytes.
    Devuelve:
    - texto extraído
    - lista de páginas que fallaron
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    paginas_texto = []
    paginas_con_error = []

    for i, pagina in enumerate(reader.pages, start=1):
        try:
            texto = pagina.extract_text()
            if texto:
                paginas_texto.append(texto)
        except Exception:
            paginas_con_error.append(i)

    texto_final = "\n\n".join(paginas_texto).strip()
    return texto_final, paginas_con_error