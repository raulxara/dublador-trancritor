"""Explicit local provisioning. Never run automatically at startup."""

import argparse
import getpass
import json
from dataclasses import asdict

from app.config.settings import Settings
from app.providers.container import Container
from app.use_cases.bootstrap_access.dtos.bootstrap_access_dto_in import BootstrapAccessDtoIn


def main():
    parser = argparse.ArgumentParser(description="Criar primeiro acesso; somente instalação vazia")
    for name in ["office-name", "office-slug", "first-name", "last-name", "email", "username"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    password = getpass.getpass("Senha (mínimo 12 caracteres): ")
    if password != getpass.getpass("Confirme a senha: "):
        parser.exit(1, "Senhas diferentes.\n")
    container = Container.build(Settings())
    try:
        result = container.bootstrap_access.exec(BootstrapAccessDtoIn(**vars(args), password=password))
        print(json.dumps(asdict(result), default=str))
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    finally:
        container.close()


if __name__ == "__main__":
    main()
