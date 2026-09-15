"""Configuração de credenciais, independente da interface."""
from collections.abc import Mapping


def resolve_api_token(secrets: Mapping, environment: Mapping) -> str:
    """Secrets têm precedência; API_TOKEN no ambiente permite execução local."""
    value = secrets.get("API_TOKEN", environment.get("API_TOKEN", ""))
    return value.strip() if isinstance(value, str) else ""
