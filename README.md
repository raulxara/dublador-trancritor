# dublador-trancritor
Aplicação que dubla textos e transcreve áudios e vídeos a partir de uma voz base. Possuo controle de velocidade da dublagem, entonação e também os áudios transcritos tem a habilidade de acompanhar o tempo e as pausas dos áudios/vídeos originais.

## Documentação da evolução para API

- [Banco de dados — modelo proposto](docs/DOC_BANCO_DE_DADOS_SIPLUG_DUBBER.html)
- [Arquitetura, tecnologias e Docker — Python no padrão SiPlug](docs/DOC_ARQUITETURA_SIPLUG_DUBBER.html)
- [Arquitetura em Markdown](docs/DOC_ARQUITETURA_SIPLUG_DUBBER.md)

A API implementa autenticação, vozes, chat e processamento assíncrono. Consulte [Worker e processamento](docs/WORKER_E_PROCESSAMENTO.md) para instalação dos modelos, migrations, resultados e limites. O banco está na revisão 0004_processing_results, com 23 tabelas de domínio.

## Executar a nova API com Docker

O ambiente de desenvolvimento e os testes rodam nos containers. Não é necessário instalar
Python, dependências ou criar uma `.venv` no host.

```sh
cd /Users/usuario/Desktop/work/ProjetoPessoal/Projects/dubber
```

Na primeira instalação, se `.env` não existir, copie `.env.example` para `.env` e defina
senhas locais distintas. Nesta implantação inicial foi criado um `.env` local com senhas
aleatórias; o arquivo é ignorado pelo Git. Não substitua essas senhas após inicializar o
volume MySQL sem atualizar também os usuários do banco.

A rede compartilhada `rede_internal` deve existir; crie com `docker network create rede_internal`
somente se ainda não existir.

```sh
docker compose build app worker scheduler
docker compose up -d --wait db
docker compose run --rm --no-deps app python -m alembic upgrade head
docker compose up -d --wait app
docker compose ps
```

Antes de iniciar worker/scheduler pela primeira vez, importe os modelos conforme [WORKER_E_PROCESSAMENTO.md](docs/WORKER_E_PROCESSAMENTO.md). Depois execute `docker compose up -d --wait worker scheduler`.

| Serviço | Acesso |
| --- | --- |
| API local | http://127.0.0.1:8087 |
| API para backends na rede SiPlug | http://siplug-dubber-api:8000 |
| MySQL local | 127.0.0.1:3323 |
| MySQL dentro dos containers | db:3306 |

As portas do host podem ser alteradas em `.env`. O banco chama-se `siplug_dubber` e o
usuário de aplicação também. Senhas ficam somente no `.env`; não são credenciais de usuário da API.

Endpoints desta etapa:

- `GET /api/v1/health`: liveness da API, sem consultar o banco.
- `GET /api/v1/ready`: conexão MySQL com `SELECT 1`; 200 quando disponível, 503 quando indisponível.

Readiness verifica conectividade, não a existência das 26 tabelas nem prontidão de modelos.
O Compose cria a base MySQL; aplique as migrations explicitamente com `docker compose exec app python -m alembic upgrade head`. Nenhuma tabela de domínio ou usuário administrativo
é criado implicitamente. Não há rotas de vozes/chat/generação expostas antes da autenticação.

```sh
docker compose run --rm --no-deps app python -m pytest -q -p no:cacheprovider
docker compose run --rm --no-deps app ruff check --no-cache app tests
docker compose run --rm --no-deps app ruff format --check app tests
docker compose logs --tail=100 app db
docker compose restart app
docker compose down
```

O bind mount atualiza os arquivos no container; reinicie `app` após editar código.
`docker compose down` preserva o volume do banco. Não acrescente `-v` se precisar conservar os dados.
A imagem DEV contém as ferramentas de teste; publicação exigirá imagem/configuração próprias.
O build instala `src/requirements.lock`, com as versões validadas dentro do Docker.
`requirements.txt` e `requirements-dev.txt` descrevem as dependências diretas; ao mudá-las,
resolva e atualize o lock dentro de Docker antes de reconstruir a imagem.

## Estrutura implementada

- `src/app/routes`, `http/controllers` e `http/dependencies`: fronteira HTTP.
- `src/app/use_cases`: casos de uso e DTOs independentes do framework.
- `src/app/entities/voice`: entidade, interface de repositório, serviço e DTOs de referência.
- `src/app/services/database`: adaptador SQLAlchemy de conectividade; consulta isolada da aplicação.
- `src/app/providers`: composição e ciclo de vida do pool de banco.
- `src/app/config`: configuração centralizada; senha com representação protegida.
- `src/app/interfaces`: contratos, incluindo Unit of Work e contratos de execução.
- `src/tests`: validação de camadas, escopo, health/readiness e guard de permissões.
- `.docker/python/Dockerfile.DEV`: imagem Python da API.
- `docker-compose.yml`: app, db, worker, scheduler, redes e persistência por volume nomeado, como no Commerce.

A dependência `RequirePermission` exige actor criado no servidor pela autenticação Bearer do router protegido. Tokens e permissões são consultados nos repositórios a cada requisição.
Não há repositório fake registrado na API. Vozes, amostras, conversas e resultados usam persistência e autorização por escritório/proprietário.

## Limpeza do legado

A interface desktop, a `.venv`, builds, executáveis, hooks PyInstaller e logs antigos
foram removidos. A aplicação em uso fica exclusivamente em `src/`, executada pelo Docker.
Dentro do container, `src/` vira `/var/www/html`.

Os motores e utilitários anteriores ficam em `legacy/audio_reference/` apenas para
consulta durante a migração; não são importados pela API nem incluídos na imagem.
Vozes, amostras e projetos em `data/` foram preservados, sem alteração ou importação automática.
Veja [referência do legado](legacy/README.md).

Pendências: reconciliação de arquivos órfãos, tags e vínculos com projetos da SiPlug.

## Autenticação persistida

As 11 tabelas de identidade estão na revisão `0001_identity`. Consulte [autenticação, bootstrap e testes](docs/AUTENTICACAO.md) para provisionar o primeiro acesso e usar as rotas protegidas. Nenhum usuário real é criado automaticamente.

## Vozes e amostras

A revisão `0002_voices` implementa as cinco tabelas de vozes/amostras e os respectivos endpoints protegidos, com arquivos no volume privado `media_data`. Consulte [rotas, envio de WAV e operação](docs/VOZES_E_AMOSTRAS.md).

## Chat, mensagens e jobs

A revisão `0003_chat_jobs` implementa conversas próprias, entrada de texto/áudio e jobs duráveis com idempotência. Os jobs entram em queued e são consumidos pelo worker implementado na revisão 0004_processing_results. Veja [rotas e responsabilidades](docs/CHAT_E_JOBS.md).

## Processamento e resultados

A revisão `0004_processing_results` implementa worker XTTS/Whisper em CPU, recuperação de leases, cancelamento em execução, transcrições e resultados privados. Consulte [responsabilidades, instalação do cache e comandos](docs/WORKER_E_PROCESSAMENTO.md).
