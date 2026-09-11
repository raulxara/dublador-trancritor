# SiPlug Dubber — Arquitetura, tecnologias e Docker

> Etapa de processamento: revisão 0004_processing_results implementa resultados, transcrições e segmentos (23 tabelas de domínio). Worker CPU com XTTS/Whisper, scheduler de leases, cancelamento em execução e download privado disponíveis. Consulte WORKER_E_PROCESSAMENTO.md para o estado atual e comandos; os registros de etapas anteriores abaixo são históricos.

> Atualização 10/09/2026: revisão 0003_chat_jobs acrescenta quatro tabelas de chat/mensagens/jobs (20 tabelas de domínio implementadas). Entrada de texto/áudio, propriedade, idempotência e cancelamento disponíveis. Jobs permanecem queued até a integração do worker. Consulte CHAT_E_JOBS.md; o restante deste documento contém especificações futuras e registros das etapas anteriores.

> Etapa de vozes implementada em 10/09/2026: revisão 0002_voices acrescenta languages, gender, voices, media_files e voice_samples, totalizando 16 tabelas de domínio. Cadastro de vozes e upload/download privado de WAV disponíveis; tags/chat/motores permanecem como alvo. Consulte VOZES_E_AMOSTRAS.md para limites e comandos atuais.

> Atualização 10/09/2026: 11 tabelas de identidade e autenticação Bearer/RBAC implementadas na revisão 0001_identity. Token persistido como SHA-256 com token_expires_at; permissions.scope_key garante unicidade global. O restante do modelo é alvo futuro. Consulte AUTENTICACAO.md para o estado atual, comandos e limitações; descrições históricas abaixo não substituem essa atualização.

> Limpeza em 10/09/2026: ambiente .venv, interface desktop, builds e empacotamento removidos. Referências históricas a app/ correspondem ao código anterior; motores preservados em legacy/audio_reference/ e dados mantidos em data/.

> Etapa inicial implementada em 10/09/2026: API Python em src/, Dockerfile.DEV, Compose app/db, health/readiness e módulo Voice de referência. O restante deste documento descreve a arquitetura alvo; worker, scheduler, autenticação completa e tabelas de domínio ainda não foram implementados. O README é a referência para executar a versão atual.

08/09/2026 • Especificação proposta para implementação • API e processamento Python

Este documento complementa [o modelo de banco de dados](DOC_BANCO_DE_DADOS_SIPLUG_DUBBER.html). A decisão confirmada pelo usuário é manter **API e processamento em Python**, reproduzindo as responsabilidades e camadas da arquitetura SiPlug. O frontend pertence à aplicação SiPlug. O Dubber será um serviço consumido pelo backend dessa aplicação.

A arquitetura completa abaixo permanece como alvo. A etapa inicial já implementa a estrutura da API e o Compose app/db. Repositórios de domínio, workers e migrations ainda serão implementados.

## 1. Fontes e estado atual

Foram analisados os docker-compose.yml de siplug-agents, siplug-notifications, siplug-logs, siplug-communications e siplug-commerce-api; o Dockerfile.DEV do Agents; a organização das APIs SiPlug trabalhadas nesta conversa; os módulos app/main.py, app/voice_manager.py, app/config.py, app/utils/projects.py, app/engines e app/audio do Dubber; as telas adicionar-voz.pdf e pagina-dublador.pdf; e o modelo das 26 tabelas documentado neste projeto.

| Tema | Encontrado no código | Destino proposto |
| --- | --- | --- |
| Interface | Desktop Python com Tkinter/CustomTkinter | Frontend somente na SiPlug; Dubber expõe API |
| Vozes | voice.json e diretórios raw/clean | Entidades voices, voice_samples e media_files |
| Resultados | Diretórios locais por data/hora | Jobs persistidos, arquivos privados e histórico de chat |
| Síntese | TTS.api, XTTS e PyTorch | Adaptador Python de síntese |
| Reconhecimento | Módulos Whisper e faster-whisper | Adaptador escolhido e validado para cada operação |
| Conversão | vc_s2s.py com síntese segmentada e ajuste de duração | Adaptador de conversão com preservação temporal configurável |
| OpenVoice | Importação opcional e placeholder | Não declarar como backend implementado |
| Persistência da API | Não implementada no Dubber atual | MySQL e migrations conforme o banco proposto |
| Docker do Dubber | Não definido por esta documentação | Topologia proposta abaixo, ainda a implementar |

Há pontos a reconciliar na implementação: asr_whisper.py importa ASR_COMPUTE_TYPE, ausente no config.py lido; as verificações de silêncio/clipping do validator não usam todos os limites declarados no config. Essas diferenças não foram corrigidas por este documento. O nome asr_openai.py não significa consumo de API remota: o módulo lido carrega Whisper localmente.

## 2. Princípios da arquitetura limpa SiPlug

Cada arquivo tem uma responsabilidade verificável. Uma classe principal por arquivo; DTO de entrada e DTO de saída em arquivos próprios. Controller não consulta banco. Caso de uso não contém SQL nem conhece Model do ORM. Repositório não decide permissão nem monta resposta HTTP. Entidade e DTO não executam rede nem carregam motores de IA.

Preservamos os conceitos de Entities, services, Dtos, interfaces de repositório e UseCases. Em Python, módulos e pastas usam snake_case; classes mantêm nomes explícitos, como VoiceEntity, IVoicesRepository, FindVoiceByUniqueIdService e SubmitDubbingMessageDtoIn. O estilo não depende da linguagem PHP: depende da responsabilidade e do sentido das dependências.

```text
Backend SiPlug
  → Rota HTTP / middleware de autenticação
  → Permissão declarada para o endpoint
  → Controller
  → DTO de entrada do caso de uso
  → Caso de uso
  → Serviços das entidades e contratos de aplicação
  → Interfaces de repositório
  → Implementações de repositório
  → MySQL
```

Domínio e aplicação não importam FastAPI, SQLAlchemy, cliente de storage ou PyTorch. Esses detalhes ficam nos adaptadores externos. O ponto de composição associa contratos às implementações e injeta as dependências por construtor; não espalha buscas globais de dependências pelos serviços.

Na proposta, a entidade representa estado e invariantes, sem persistência automática. O serviço da entidade coordena sua operação e chama o repositório. Isso evita acoplamento Entity → Repository → Entity. O padrão legado Entity.create() encontrado em outras APIs não é reproduzido automaticamente: preservar a divisão de responsabilidades tem prioridade sobre copiar esse acoplamento.

## 3. Responsabilidade de cada camada

| Camada / arquivo | Responsabilidade | Não deve fazer |
| --- | --- | --- |
| routes/api.py | Declarar endpoints, schemas HTTP e permissões | Executar fluxos de negócio dentro da rota |
| resolve_actor_authorization_middleware.py | Validar Bearer, estados e criar actor | Aceitar officeId ou permissões do JSON como autoridade |
| require_permission_dependency.py | Exigir a permissão da rota antes do controller | Implementar bypass pelo nome do cargo |
| authorized_actor.py | Contexto imutável de usuário, vínculo, escritório e permissões | Transportar token ou senha para o caso de uso |
| Controller | Adaptar HTTP para DTO; chamar caso de uso; mapear saída | Abrir sessão SQL, chamar ORM ou gerar áudio |
| HTTP schema | Validar forma HTTP e nomes JSON, como voiceId | Conter regra de autorização ou consultar banco |
| UseCaseDtoIn | Contrato imutável da operação, independente do transporte | Receber Request, ORM Model ou sessão de banco |
| UseCaseDtoOut | Resultado da operação | Expor segredo, caminho interno ou dependência HTTP |
| UseCaseService | Coordenar serviços, autorização de recursos e transação | Conter SELECT/UPDATE, FFmpeg ou acesso direto ao filesystem |
| Entity | Estado e invariantes locais do domínio | Ler request, env, banco ou modelo de IA |
| Entity service | Operação pequena e reutilizável sobre uma entidade | Controlar todo o fluxo de chat e processamento |
| Entity service DTOs | Contratos de entrada e saída da operação da entidade | Repassar parâmetros indefinidos sem validação |
| IRepository | Contrato de persistência com escopo explícito | Retornar Query, Session ou Model do ORM |
| Repository | Consultas, persistência e mapeamento para domínio | Descobrir usuário autenticado ou montar JSON HTTP |
| ORM Model | Mapear tabelas, colunas, índices e relações | Conter fluxo completo de dublagem |
| Resource authorization service | Verificar escritório, proprietário e associações | Tratar existência do ID como autorização |
| IUnitOfWork | Contrato de transação dos casos de uso | Expor SQL ao domínio |
| SqlAlchemyUnitOfWork | Gerenciar sessão, commit e rollback | Abrir transação durante inferência longa |
| IMediaStorage / adaptador | Gravar, verificar e acessar arquivos privados | Aceitar caminhos arbitrários do consumidor |
| IAudioEngine / adaptador | Executar síntese, reconhecimento ou conversão | Conhecer usuário HTTP ou montar resposta de chat |
| Worker entrypoint | Obter dependências e executar caso de uso de processamento | Duplicar regras dos casos de uso |
| Scheduler entrypoint | Executar recuperação e limpeza controlada | Alterar tabelas diretamente fora dos serviços |
| providers/container.py | Montar dependências para API, worker e scheduler | Rodar negócio durante importação |
| Exception handler | Mapear erros de domínio para HTTP e logs | Converter indiscriminadamente qualquer erro em 500 |
| Migration | Evoluir schema e restrições | Criar credenciais reais automaticamente |

DTOs validam invariantes locais: ID não vazio, tipo, intervalo e combinação texto/arquivo. Existência, estado, escritório e propriedade pertencem aos serviços. No FastAPI, validação por Pydantic fica no transporte; DTOs internos podem ser dataclasses imutáveis. Não criar cópias de dados sem finalidade: schemas HTTP justificam aliases/validação do transporte, DTOs justificam independência do caso de uso.

## 4. Estrutura de diretórios proposta

```text
dubber/
├── docker-compose.yml                  # infraestrutura local futura
├── .docker/
│   ├── python/
│   │   ├── Dockerfile.API.DEV           # API, banco e dependências leves
│   │   └── Dockerfile.WORKER.DEV        # worker, FFmpeg e modelos Python
│   └── mysql/data/                      # persistência local, fora do Git
├── src/
│   ├── app/
│   │   ├── main.py                      # composição da aplicação HTTP
│   │   ├── routes/api.py
│   │   ├── http/
│   │   │   ├── controllers/submit_dubbing_message_controller.py
│   │   │   ├── schemas/submit_dubbing_message_request.py
│   │   │   ├── middleware/resolve_actor_authorization_middleware.py
│   │   │   ├── dependencies/require_permission_dependency.py
│   │   │   └── exception_handlers.py
│   │   ├── entities/
│   │   │   ├── voice/
│   │   │   │   ├── voice_entity.py
│   │   │   │   ├── i_voices_repository.py
│   │   │   │   └── services/find_voice_by_unique_id/
│   │   │   │       ├── find_voice_by_unique_id_service.py
│   │   │   │       └── dtos/
│   │   │   │           ├── find_voice_by_unique_id_dto_in.py
│   │   │   │           └── find_voice_by_unique_id_dto_out.py
│   │   │   ├── dubbing_chat/
│   │   │   ├── dubbing_message/
│   │   │   ├── dubbing_job/
│   │   │   └── ...                      # demais entidades documentadas
│   │   ├── use_cases/submit_dubbing_message/
│   │   │   ├── submit_dubbing_message_use_case_service.py
│   │   │   └── dtos/
│   │   │       ├── submit_dubbing_message_dto_in.py
│   │   │       └── submit_dubbing_message_dto_out.py
│   │   ├── interfaces/
│   │   │   ├── i_unit_of_work.py
│   │   │   ├── i_media_storage.py
│   │   │   └── i_audio_engine.py
│   │   ├── services/
│   │   │   ├── voice/voices_repository.py
│   │   │   ├── dubbing_job/dubbing_jobs_repository.py
│   │   │   ├── actor/authorized_actor.py
│   │   │   ├── authorization/require_dubbing_resource_access_service.py
│   │   │   ├── transactions/sqlalchemy_unit_of_work.py
│   │   │   ├── storage/local_media_storage.py
│   │   │   └── audio_engine/xtts_audio_engine.py
│   │   ├── models/                     # modelos SQLAlchemy
│   │   ├── audio/                      # validação, normalização e exportação
│   │   ├── engines/                    # motores atuais adaptados
│   │   ├── workers/main.py
│   │   ├── scheduler/main.py
│   │   ├── exceptions/
│   │   ├── config/settings.py
│   │   └── providers/container.py
│   ├── database/migrations/            # Alembic
│   ├── tests/unit/
│   ├── tests/integration/
│   ├── tests/http/
│   ├── pyproject.toml
│   └── requirements.lock               # nome proposto; gerador a definir
└── docs/
    ├── DOC_BANCO_DE_DADOS_SIPLUG_DUBBER.html
    ├── DOC_ARQUITETURA_SIPLUG_DUBBER.html
    └── DOC_ARQUITETURA_SIPLUG_DUBBER.md
```

A árvore representa o alvo completo; a etapa inicial já implementa parte das pastas e contratos. A interface desktop foi removida na limpeza. Os motores anteriores estão em legacy/audio_reference/ apenas para consulta e migração; o entrypoint HTTP está em src/app/main.py. A API não importa engines/ ou PyTorch no boot; somente o ponto de composição do worker carrega esses adaptadores.

## 5. Identificadores, entidades, DTOs e repositórios

Regra do banco: id numérico é técnico; _id é VARCHAR(255) e identifica o domínio. Todo *_id de relacionamento usa VARCHAR(255) e referencia _id do destino. Nenhum contrato externo usa o id numérico.

No SQL e nos atributos Python: office_id, voice_sample_id, input_message_id. Nos contratos JSON da SiPlug: officeId, voiceSampleId, inputMessageId. Schemas de transporte fazem o mapeamento; o caso de uso não conhece aliases HTTP. A classe pode representar a coluna _id como unique_id para evitar nomes privados ambíguos; esse valor continua sendo exclusivamente o _id do banco.

Exemplo ilustrativo, em arquivos separados:

```python
# entities/voice/i_voices_repository.py
from typing import Protocol
from .voice_entity import VoiceEntity

class IVoicesRepository(Protocol):
    def find_by_unique_id(self, office_id: str, unique_id: str) -> VoiceEntity | None:
        ...
```

FindVoiceByUniqueIdDtoIn carrega office_id e unique_id. FindVoiceByUniqueIdService consulta IVoicesRepository e devolve seu DTO de saída. VoicesRepository usa os dois filtros e converte ORM em domínio. Não devolve consulta incompleta para o controller filtrar depois.

SubmitDubbingMessageDtoIn carrega actor, chat_id, operation, voice_id, text ou media_file_id, parâmetros e idempotency_key. O caso de uso chama serviços para resolver chat, arquivo, voz e amostra pronta; grava o snapshot no job. SubmitDubbingMessageDtoOut devolve chat_id, message_id, job_id e processing_state. Token e storage_key não integram a resposta.

IUnitOfWork delimita a transação. Os repositórios envolvidos compartilham a mesma sessão daquela unidade de trabalho; não abrem conexões independentes ou fazem commits por conta própria. O mapeamento de exceptions preserva erro de autorização, validação e conflito, sem mascará-los como falha genérica. Sessões devem ter ciclo de vida e transação explícitos, conforme a [documentação de sessões do SQLAlchemy](https://docs.sqlalchemy.org/en/20/orm/session_basics.html).

## 6. Tecnologias utilizadas e propostas

| Componente | Tecnologia | Situação |
| --- | --- | --- |
| Linguagem da API e worker | Python | Decisão confirmada; motores atuais já são Python |
| HTTP da API | FastAPI + Uvicorn | Proposto, ainda não implementado |
| Schemas HTTP e configuração | Pydantic / pydantic-settings | Proposto; restrito ao transporte/configuração |
| DTOs internos | dataclasses imutáveis | Proposto; domínio independente do HTTP |
| Contratos | Protocol ou ABC | Proposto; interfaces explícitas para repositório, storage e motores |
| ORM e transações | SQLAlchemy | Proposto, restrito à persistência e Unit of Work |
| Driver MySQL | Driver compatível com a abordagem síncrona inicial, como PyMySQL | Proposto; versão a fixar com o conjunto |
| Migrations | Alembic | Proposto; implementação das 26 tabelas documentadas |
| Banco | MySQL / InnoDB | Referências usam 8.0 e 8.4; proposta segue família 8.4 do Commerce |
| Fila inicial | dubbing_jobs no MySQL | Persistência durável e reivindicação pelo worker; não exige Redis na primeira versão |
| Síntese | PyTorch + TTS/XTTS | Encontrado no código atual |
| Transcrição | Whisper/faster-whisper | Encontrado; escolher adaptador e testar suporte a timestamps |
| Áudio | FFmpeg, NumPy, SoundFile, librosa | Encontrado no código atual |
| Arquivos | Volume privado compartilhado | Proposto inicialmente; adaptador permite evolução para storage externo |
| Testes | pytest | Proposto, com testes unitários, HTTP, persistência e motores |
| Qualidade | Ruff e verificação de tipos, como mypy | Proposto; configuração e versões devem ser fixadas |
| Dependências | pyproject.toml e lock reproduzível | Proposto; não copiar a .venv local para a imagem |

Versões exatas de Python, bibliotecas e imagens devem ser congeladas após validar o conjunto com os motores atuais. Esta documentação não declara compatibilidade de versões ainda não testadas. O arquivo de lock deve identificar dependências da API e do worker; a imagem da API não precisa carregar bibliotecas pesadas de inferência.

CPU é a referência inicial para validar o fluxo. GPU depende do host, runtime e imagens apropriadas; não pressupor que o Docker Linux terá acesso ao GPU do Mac. Selecionar outro idioma não implementa tradução automaticamente. OpenVoice permanece fora da lista de recursos implementados.

## 7. Padrão Docker observado nas referências

| Projeto | HTTP no host | Banco no host | Características |
| --- | --- | --- | --- |
| Agents | 8084 → 80 | 3322 → 3306 | app, db, .docker/php/Dockerfile.DEV, src em /var/www/html |
| Notifications | 8081 → 80 | 3318 → 3306 | app, db, docker/php/Dockerfile.DEV e local.ini |
| Logs | 8080 → 81 | 3319 → 3306 | app, db; porta interna HTTP diferente |
| Communications | 8083 → 80 | 3321 → 3306 | app, db e .docker/php/Dockerfile.DEV |
| Commerce | Nginx 8086 → 80 | 3307 → 3306 | PHP-FPM, Nginx, MySQL 8.4, Redis, MinIO, worker e scheduler |

Elementos preservados no Dubber: serviço app, código em ./src, working_dir /var/www/html, Dockerfiles dentro de .docker, banco MySQL persistente, rede local do projeto e rede compartilhada rede_internal. Worker e scheduler separados seguem a organização do Commerce, mas executam Python.

Não copiar PHP-FPM, Composer, Apache ou local.ini para a aplicação Python. A API usa Uvicorn em 8000. O gateway/reverse proxy de publicação pode ser fornecido pela infraestrutura SiPlug; um Nginx próprio é opcional. A porta interna não precisa ser 80 para preservar o padrão arquitetural.

Os arquivos antigos declaram version: '3.2'; o Commerce não declara version. O exemplo proposto segue a segunda forma. Não copiar senhas demonstrativas das referências como credenciais de implantação.

## 8. Serviços, redes e volumes do Dubber

| Serviço | Papel | Redes | Persistência e portas |
| --- | --- | --- | --- |
| app | API FastAPI/Uvicorn | local + rede_internal | ./src e mídia privada; host 8087 → 8000, sugestão |
| db | MySQL do Dubber | local | ./.docker/mysql/data:/var/lib/mysql; host 3323 → 3306, opcional |
| worker | Reivindicar jobs e gerar áudio | local | ./src, mídia privada e cache de modelos; sem porta pública |
| scheduler | Recuperar concessões expiradas e limpar órfãos com segurança | local | ./src e mídia privada; sem porta pública |

8087 e 3323 não aparecem nos cinco Compose comparados, mas não houve inspeção das portas realmente em uso no computador. Confirmar disponibilidade antes de implementar. O backend dentro da rede compartilhada usa http://siplug-dubber-api:8000/api, não localhost:8087. MySQL usa db:3306 dentro dos containers.

A rede local não é compartilhada deliberadamente com as outras APIs. Isso não equivale à opção Docker internal: true; política de saída para baixar modelos/integrações deve ser definida. rede_internal é uma rede externa que precisa existir. O alias siplug-dubber-api evita depender do nome genérico app usado em várias aplicações.

Mídia em /var/lib/dubber/media, fora da raiz do código e de qualquer diretório público. Modelos em /var/cache/dubber/models. O servidor entrega arquivos somente após autorização. API e worker compartilham os mesmos caminhos; UID/GID e permissões do volume precisam ser compatíveis, sem usar chmod 777 como estratégia.

## 9. Esqueleto Compose proposto

Exemplo documental, ainda não executável no projeto: faltam os Dockerfiles, a API, o worker, o scheduler, configuração e dependências. Não salvar na raiz e esperar que o desktop atual se transforme em API automaticamente.

```yaml
name: siplug-dubber

x-runtime: &runtime
  working_dir: /var/www/html
  env_file:
    - ./src/.env
  volumes:
    - ./src:/var/www/html
    - media_data:/var/lib/dubber/media
  depends_on:
    db:
      condition: service_healthy
  networks: [local]

services:
  app:
    <<: *runtime
    build:
      context: .
      dockerfile: ./.docker/python/Dockerfile.API.DEV
    command: [python, -m, uvicorn, app.main:app, --host, 0.0.0.0, --port, "8000"]
    ports:
      - "127.0.0.1:${DUBBER_HTTP_PORT:-8087}:8000"
    healthcheck:
      test: [CMD, python, -c, "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      local:
      rede_internal:
        aliases: [siplug-dubber-api]

  db:
    image: ${DUBBER_MYSQL_IMAGE:?Defina uma imagem MySQL 8.4 validada}
    restart: unless-stopped
    environment:
      MYSQL_DATABASE: siplug_dubber
      MYSQL_USER: siplug_dubber
      MYSQL_PASSWORD: ${DUBBER_DB_PASSWORD:?Defina a senha da aplicacao}
      MYSQL_ROOT_PASSWORD: ${DUBBER_DB_ROOT_PASSWORD:?Defina a senha administrativa}
      TZ: UTC
    ports:
      - "127.0.0.1:${DUBBER_DB_PORT:-3323}:3306"
    volumes:
      - ./.docker/mysql/data:/var/lib/mysql
    healthcheck:
      test: ["CMD-SHELL", 'MYSQL_PWD="$${MYSQL_PASSWORD}" mysql -h 127.0.0.1 -u "$${MYSQL_USER}" "$${MYSQL_DATABASE}" -e "SELECT 1" >/dev/null 2>&1']
      interval: 5s
      timeout: 5s
      retries: 20
    networks: [local]

  worker:
    <<: *runtime
    build:
      context: .
      dockerfile: ./.docker/python/Dockerfile.WORKER.DEV
    command: [python, -m, app.workers.main]
    init: true
    stop_grace_period: 16m
    volumes:
      - ./src:/var/www/html
      - media_data:/var/lib/dubber/media
      - model_cache:/var/cache/dubber/models

  scheduler:
    <<: *runtime
    build:
      context: .
      dockerfile: ./.docker/python/Dockerfile.API.DEV
    command: [python, -m, app.scheduler.main]

volumes:
  media_data:
  model_cache:

networks:
  local:
  rede_internal:
    external: true
```

Dockerfile.API.DEV: imagem Python de versão fixada, diretório /var/www/html, instalação das dependências leves a partir do lock e usuário sem privilégios de root para executar a aplicação. Dependências devem ser instaladas fora do diretório substituído pelo bind mount; por exemplo, ambiente em /opt/venv.

Dockerfile.WORKER.DEV: mesma base Python compatível, dependências de persistência e domínio, bibliotecas dos motores, FFmpeg e libsndfile quando necessário. Cache de modelos separado. Não baixar modelos em cada job. Dependências e pesos de modelo precisam de versões identificáveis; modelo não disponível significa worker sem prontidão para processar.

Não montar a .venv do macOS no Linux. Ignorar .git, .venv, dados, logs e segredos no contexto de build por .dockerignore. Bind mounts são configuração de desenvolvimento; em publicação, construir imagens imutáveis com o código e implantar configuração/segredos externamente.

A relação entre depends_on e service_healthy segue a [documentação de inicialização do Docker Compose](https://docs.docker.com/compose/how-tos/startup-order/). Healthcheck da API verifica o endpoint proposto. Não prova que o worker ou modelo estão prontos. Implementar verificação separada de worker e supervisão dos processos; não anunciar prontidão de inferência apenas porque HTTP responde. O stop_grace_period é um valor inicial, a alinhar com os limites reais dos jobs e do ambiente de publicação.

## 10. Configuração centralizada

| Arquivo / variável | Uso proposto |
| --- | --- |
| .env na raiz do Compose | Imagem MySQL, portas e senhas usadas na interpolação do Compose |
| src/.env | Configuração dos processos Python; não versionar valores reais |
| app/config/settings.py | Ler e validar ambiente uma vez; injetar configuração nos adaptadores |
| DB_HOST / DB_PORT | db / 3306 |
| DB_DATABASE / DB_USERNAME | siplug_dubber / siplug_dubber |
| DB_PASSWORD | Mesmo valor configurado no Compose para o usuário da aplicação |
| DUBBER_MEDIA_ROOT | /var/lib/dubber/media |
| DUBBER_MODEL_CACHE | /var/cache/dubber/models |
| DUBBER_JOB_POLL_SECONDS | Intervalo de consulta de jobs; proposta inicial 2 segundos |
| DUBBER_JOB_TIMEOUT_SECONDS | Prazo máximo de processamento; definir com limites de entrada e medições |
| DUBBER_WORKER_CONCURRENCY | Proposta inicial 1 por worker; aumentar somente após medir memória/CPU |
| DUBBER_WORKER_LEASE_SECONDS | Prazo de posse do job; renovado durante execução válida |
| DUBBER_MAX_JOB_ATTEMPTS | Limite de tentativas antes de falha definitiva |

Nomes DUBBER_* são propostos, não variáveis já implementadas. env_file injeta ambiente no container; não alimenta a interpolação ${...} do Compose. Manter arquivos .env.example sem segredos. Não ler os.environ dentro de entidades, DTOs e casos de uso. Versões de modelo e limites configurados devem ficar registrados no snapshot do job.

Não armazenar Bearer administrativo em frontend. Credenciais de integração externas permanecem no backend. Logs de operação incluem IDs de correlação, estado e duração, sem tokens, texto integral, áudios ou dados sensíveis. A imagem da API deve falhar na inicialização se faltar configuração obrigatória.

## 11. Processamento assíncrono com banco como fila inicial

A API recebe a solicitação e devolve 202 com jobId. Mensagem, entrada e job são criados em transação curta. Não é necessária uma publicação adicional em broker para tornar o trabalho visível: o worker consulta dubbing_jobs. Isso evita introduzir Redis/Celery antes de haver necessidade demonstrada.

O caso de uso ClaimNextDubbingJob usa um serviço de entidade e método de repositório para selecionar um job queued de forma concorrente segura. O repositório pode implementar seleção com bloqueio e SKIP LOCKED no MySQL escolhido, atualizar processing_state, worker_token, locked_until e attempts e encerrar a transação. Nenhum SELECT ou UPDATE fica no entrypoint do worker ou no caso de uso.

Depois da posse, o worker obtém arquivos e chama IAudioEngine fora da transação. A interface recebe entrada autorizada, amostra selecionada, parâmetros e deadline. A implementação invoca os motores Python atuais. Não é necessário um servidor HTTP interno de engine separado: a separação inicial é de processo e de camadas, com API leve e worker dedicado.

O processamento pesado não pode bloquear o event loop da API. Para a primeira implementação, sessões de banco síncronas e endpoints síncronos podem ser usados de forma coerente; não declarar tudo async e executar ORM síncrono/inferência bloqueante dentro do event loop. Essa distinção acompanha a [documentação de concorrência do FastAPI](https://fastapi.tiangolo.com/async/).

Durante a execução, renovar a posse por mecanismo que continue funcionando mesmo durante inferência bloqueante. Supervisionar processamento em subprocesso ou mecanismo equivalente para aplicar timeout/cancelamento. Cada tentativa escreve em diretório próprio. Um worker cuja posse expirou não pode publicar resultado: a atualização final precisa comparar worker_token e estado vigente.

O scheduler recupera jobs com concessão expirada usando serviços/repositórios e atualizações condicionais. Recoloca em queued quando permitido ou marca failed ao exceder limite. Começar com uma instância; múltiplas instâncias exigem exclusão mútua no banco. Não recuperar um job com heartbeat válido. Não apagar arquivos temporários de execuções ativas.

A conclusão registra arquivos, outputs, resposta assistant e vínculos em transação idempotente. Arquivos precisam existir antes de marcar completed. Banco e filesystem não compartilham transação: limpar órfãos e reparar publicações incompletas por rotinas controladas. Os estados da documentação do banco continuam: status active/inactive; processing_state queued/processing/completed/failed/cancelled.

Uma fila externa pode ser introduzida depois via contrato. Redis/Celery seriam opções futuras, não serviços obrigatórios desta versão. Se adicionados, definir formato da mensagem, publicação após commit, recuperação de falhas entre banco e broker e deduplicação. O banco permanece a fonte do estado de negócio.

## 12. Exemplo completo: texto ou áudio no chat

1. Middleware valida Authorization: Bearer contra user_customers, users e offices ativos; forma AuthorizedActor no servidor.
2. A dependência da rota exige dubbing.generate. O controller valida o schema HTTP e monta SubmitDubbingMessageDtoIn com o actor.
3. O caso de uso consulta chat, voz e arquivo por serviços; verifica escritório, proprietário e amostra pronta. Seleção enviada é identificador de recurso, não identidade autorizada.
4. Para texto, captura input_text e operation=text_to_speech. Para áudio, captura input_file_id e operation=speech_to_speech. Cada execução guarda voice_sample_id, idioma, velocidade, pitch, preserve_timing e parâmetros.
5. Em transação, cria mensagem, anexos e job. A combinação office_id + user_customer_id + idempotency_key é única. Mesmo conteúdo normalizado retorna o job existente; conteúdo diferente na mesma chave retorna conflito. Tratar também a violação concorrente do índice único.
6. A API retorna 202. O frontend da SiPlug acompanha o job por chamadas ao próprio backend, que consulta o Dubber.
7. O worker reivindica o job e usa o motor adequado. Se houver reconhecimento, grava transcrição e segmentos por serviços de domínio/persistência após o processamento.
8. O resultado cria media_files, dubbing_job_outputs, mensagem assistant/result e anexos. WAV e MP3 podem coexistir.
9. Download é autorizado por escritório e propriedade. Salvar no projeto cria project_audio_links após a SiPlug validar o projeto externo.

Modificar a voz selecionada no chat não muda a amostra registrada em jobs antigos. Alterar texto já aceito gera nova mensagem/execução ou versão explícita, sem reescrever o histórico.

## 13. Contratos HTTP propostos

| Endpoint | Resultado / controle |
| --- | --- |
| GET /api/v1/health | Verificação básica sem dados sensíveis |
| GET /api/v1/auth/context | Contexto calculado pelo Bearer |
| POST /api/v1/media-files | Upload autorizado; validar formato, tamanho e duração |
| GET /api/v1/voices | Catálogo do escritório, com filtros e paginação |
| POST /api/v1/voices | Cadastrar voz e solicitar preparação de amostra |
| POST /api/v1/dubbing-chats | Criar conversa do vínculo autenticado |
| POST /api/v1/dubbing-chats/{chatId}/messages | Texto ou arquivo cadastrado; criar job idempotente |
| GET /api/v1/dubbing-chats/{chatId}/messages | Histórico autorizado e paginado |
| GET /api/v1/dubbing-jobs/{jobId} | Estado e resultados autorizados |
| POST /api/v1/dubbing-jobs/{jobId}/cancel | Solicitar cancelamento autorizado |
| GET /api/v1/media-files/{fileId}/download | Download protegido ou URL temporária |

São contratos propostos, não rotas existentes. Upload e geração separados evitam retransmitir o arquivo a cada retentativa. Respostas: 401 identidade inválida; 403 permissão ausente; 404 recurso indisponível no escopo; 409 conflito de estado/idempotência; 422 entrada inválida; 202 processamento aceito.

Definição canônica de segurança permanece no documento de banco: user_customers.token autentica entrada; api_credentials.token guarda credencial externa. Biblioteca HTTP, CORS, rede Docker e frontend não substituem a validação no servidor. Política de Origin adicional, se adotada, deve contemplar as chamadas do backend e não ser tratada como identidade.

## 14. Testes, implantação e migração

Unitários: DTOs, invariantes, seleção de operação, autorização de recursos e casos de uso com contratos substituíveis. Integração: repositórios reais, filtros office_id, FKs para _id, Unit of Work e concorrência de jobs. HTTP: 401/403, contexto falsificado, propriedade, upload/download e idempotência. Áudio: amostras pequenas para validação/normalização; inferência real em conjunto separado de testes.

Cenários essenciais: texto → voz; áudio → voz selecionada; troca de voz sem alterar histórico; transcrição segmentada; trabalhador interrompido; resultado duplicado; cancelamento; falha de armazenamento; tentativa de outro escritório consultar arquivo/job; amostra rejeitada; modelo indisponível. Usar banco MySQL isolado para validar bloqueios e semântica de concorrência; SQLite sozinho não comprova esses comportamentos.

Plano de implementação: congelar versões compatíveis; criar estrutura e imagens; implementar migrations das 26 tabelas; autenticação e permissões; cadastro/validação de vozes; chat e jobs; adaptação dos motores; worker/recuperação; importação legada com mapa de _id; integração das telas SiPlug; retirada do desktop após equivalência funcional.

Na implantação futura: criar rede_internal se ausente; validar docker compose config; construir imagens; verificar readiness; executar migrations Alembic explicitamente; provisionar primeiro acesso pelo fluxo autorizado; testar consumidor real em homologação. Não executar limpeza de banco ou bootstrap indiscriminado em instalação existente. Manter backup de mídia, banco e configuração necessária à recuperação.

Atualização: a etapa inicial criou Dockerfile.DEV, docker-compose.yml e API FastAPI com health/readiness, validados em containers com MySQL. Worker, scheduler, migrations de domínio e endpoints de negócio continuam pendentes. O exemplo acima é a arquitetura alvo; consulte o Compose da raiz e o README para a execução atual. Motores e desktop foram preservados.
