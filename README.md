# llm-text-lab

Laboratorio de aprendizaje en Python para construir una herramienta de análisis documental con LLMs, recuperación semántica y memoria conversacional básica.

## Qué hace el proyecto

Este proyecto permite trabajar con documentos `.txt` desde consola y realizar distintas acciones apoyadas en modelos de OpenAI:

- análisis de texto por modos
- preguntas semánticas sobre documentos
- recuperación de fragmentos relevantes mediante embeddings
- reutilización de caché vectorial por documento
- memoria conversacional reciente dentro de la sesión
- historial de preguntas y exportación a texto

## Funcionalidades actuales

- Selección de documentos desde la carpeta `data/`
- División automática del texto en fragmentos
- Modos de análisis:
  - `resumen`
  - `puntos_clave`
  - `clasificacion`
  - `tono`
  - `todos`
- Modo `pregunta_semantica` con:
  - recuperación por similitud semántica
  - múltiples preguntas seguidas sobre el mismo documento
  - memoria conversacional reciente
- Caché vectorial por documento en `cache/`
- Validación de caché mediante hash del documento
- Historial de sesión en memoria
- Exportación del historial a `exports/`

## Estructura del proyecto

```text
llm-text-lab/
│
├── app/
│   ├── config.py
│   ├── embeddings.py
│   ├── main.py
│   ├── prompts.py
│   └── utils.py
│
├── data/
│   ├── ejemplo.txt
│   └── oferta.txt
│
├── cache/          # ignorado por Git
├── exports/        # ignorado por Git
├── .env            # ignorado por Git
├── .gitignore
├── README.md
└── requirements.txt