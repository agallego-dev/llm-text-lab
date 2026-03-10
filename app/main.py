from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.utils import leer_texto


def main() -> None:
    texto = leer_texto("data/ejemplo.txt")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
    Lee el siguiente texto y devuelve:
    1. Un resumen breve
    2. Tres puntos clave
    3. Una clasificación del tema principal

    Texto:
    {texto}
    """

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    print("=== RESPUESTA DEL MODELO ===")
    print(response.output_text)


if __name__ == "__main__":
    main()