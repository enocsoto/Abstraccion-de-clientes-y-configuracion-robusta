# Abstracción de clientes y configuración robusta

Una fábrica elige el cliente de OpenAI o Anthropic. Las claves quedan en `SecretStr` y las dos respuestas salen como texto plano.

## Cómo correrlo

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
```

Completa la clave del proveedor y ejecuta:

```bash
.venv/bin/python ejercicio.py
```

`ejercicio.py` es el archivo único del ejercicio. El paquete `llm` parte la misma solución para los tests.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
