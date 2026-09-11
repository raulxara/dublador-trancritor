# Referência dos motores de áudio

`audio_reference/` preserva o código de processamento da aplicação desktop para migração futura aos services/adaptadores e workers da API.

Este diretório não é executável, não faz parte da imagem Docker e não deve ser importado pela API. Imports antigos `app.*`, caminhos locais e inicialização das bibliotecas permanecem apenas como referência; precisarão ser adaptados antes de reutilização. Não coloque este diretório no PYTHONPATH.

A interface Tkinter, os hooks PyInstaller, os executáveis, a `.venv` e os builds foram removidos. O código desktop versionado pode ser consultado no histórico Git. Vozes, amostras e projetos continuam preservados em `data/`; ainda não foram importados para o banco.
