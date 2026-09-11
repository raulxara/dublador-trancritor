# Worker e processamento — SiPlug Dubber

> Atualização 11/09/2026: revisão 0005_catalog_and_projects completa as 26 tabelas de domínio. Tags, vínculos externos de projetos, edição versionada de transcrições, entrada MP3/MP4 e limpeza protegida de órfãos implementados. Consulte CATALOGO_PROJETOS_E_MIDIA.md para contratos e limites atuais; registros anteriores abaixo são históricos.

## Estado implementado

A revisão `0004_processing_results` acrescenta `dubbing_job_outputs`, `transcriptions` e `dubbing_segments`: **23 tabelas de domínio**, além de `alembic_version`. Todas as referências internas continuam sendo VARCHAR(255) para `_id`; relações privadas incluem office_id. `status` continua active/inactive; o andamento usa processing_state. Tags, voice_tags e project_audio_links foram implementados na revisão 0005_catalog_and_projects.

O Compose possui API, MySQL, worker e scheduler. A API recebe a solicitação e responde 202. O worker executa os modelos reais em CPU, usando somente arquivos locais. O scheduler recupera tentativas abandonadas. Nenhum processamento de modelo ocorre no processo HTTP ou dentro de uma transação do banco.

Operações:

- `text_to_speech`: texto → XTTS v2 condicionado pela amostra selecionada → WAV.
- `transcribe`: áudio → Whisper tiny → TXT, transcrição e segmentos.
- `speech_to_speech`: áudio → Whisper → síntese XTTS com a voz selecionada → WAV e TXT. Não é tradução nem conversão direta do sinal original; o reconhecimento pode alterar palavras, e a prosódia pode mudar.

## Responsabilidade de cada arquivo

Os caminhos abaixo são relativos a `src/app/`.

| Arquivo ou família | Responsabilidade |
| --- | --- |
| entities/dubbing_job/*_entity.py | Contratos imutáveis de posse, entrada, segmento, arquivo e resultado; sem ORM ou motores |
| entities/dubbing_job/services/claim_job | Validar política e gerar token privado de execução |
| entities/dubbing_job/services/renew_job | Renovar posse por meio do contrato de repositório |
| entities/dubbing_job/services/recover_jobs | Validar política de recuperação |
| entities/dubbing_job/services/fail_job | Aceitar somente códigos sanitizados de falha |
| entities/dubbing_job/services/resolve_execution | Obter snapshot autorizado da entrada |
| entities/dubbing_job/services/complete_execution | Encaminhar publicação para o repositório transacional |
| entities/dubbing_job/services/cancel_active_job | Cancelar queued/processing; recusar estados finais incompatíveis |
| entities/dubbing_job/services/list_job_outputs | Consultar resultados por escritório e job |
| */dtos/*_dto_in.py e *_dto_out.py | Entrada e saída próprias, em arquivos separados |
| interfaces/i_*execution*.py e i_job_outputs_repository.py | Portas da aplicação; sem SQLAlchemy, FastAPI, PyTorch ou filesystem |
| models/processing/sqlalchemy_job_execution_repository.py | SQL atômico de claim, heartbeat, falha e recuperação |
| models/processing/sqlalchemy_execution_results_repository.py | Consultar relações persistidas e publicar mensagem/arquivos/transcrição/segmentos/estado numa única transação |
| models/processing/sqlalchemy_job_outputs_repository.py | Consultar apenas resultados e arquivos ativos de jobs concluídos |
| use_cases/process_next_job | Orquestrar claim → resolução → motor → publicação/falha; sem SQL nem processos do SO |
| use_cases/recover_expired_jobs | Orquestrar serviço de recuperação |
| use_cases/list_dubbing_outputs e download_dubbing_output | Validar permissão, proprietário e conversa antes de devolver metadados/caminho privado ao controller |
| services/processing/subprocess_audio_execution.py | Criar diretório privado da tentativa; supervisionar subprocesso, heartbeat e timeout; validar manifest |
| services/processing/private_result_storage.py | Validar WAV/TXT, gerar chave privada, persistir arquivo e checksum |
| engines/local_audio_engine.py | Adaptador concreto de XTTS/Whisper; carregar checkpoints e executar operações |
| engines/audio_filters.py | Executar FFmpeg com argumentos explícitos para velocidade, pitch e duração |
| engines/run_attempt.py | Entrada do subprocesso; ler pedido local e escrever manifest de resultados |
| providers/database_engine_factory.py | Criar engine SQLAlchemy com configuração centralizada e parâmetros ocultos |
| providers/worker_container.py e scheduler_container.py | Compor dependências de cada processo |
| workers/main.py e scheduler/main.py | Laço de execução, sinais, espera e descarte de recursos |
| workers/health.py | Heartbeat local para healthcheck Docker, sem porta HTTP adicional |
| console/import_audio_models.py | Importação explícita de checkpoints existentes para volume, com manifest SHA-256 |
| http/controllers/chat/*output*_controller.py | HTTP → DTO → caso de uso → JSON/FileResponse |

Os repositórios da fila abrem transações curtas por operação. O repositório de publicação mantém sua própria transação única. Os casos de uso de leitura reaproveitam o ChatUnitOfWork. Não há conexão global compartilhada entre tentativas.

## Posse, recuperação e cancelamento

1. `SELECT ... FOR UPDATE SKIP LOCKED` escolhe o job queued mais antigo elegível. O claim grava processing, token aleatório de 64 caracteres, attempts e locked_until na mesma transação.
2. O worker resolve **voice_sample_id capturado no job**, não a amostra atual da voz. Revalida escritório, usuário, proprietário, perfil, conversa, mensagem, arquivos, voz, amostra e idioma ativos.
3. O subprocesso recebe caminhos privados e parâmetros, sem senha do banco nem token de autenticação. Bibliotecas pesadas são importadas somente nele.
4. O supervisor renova a posse a cada 10 segundos; a concessão dura 120 segundos. Posse vencida não pode ser renovada. Tempo máximo padrão: 1200 segundos por subprocesso.
5. Cancelamento queued/processing invalida worker_token e locked_until. O supervisor detecta a perda no heartbeat e encerra o grupo de processos, incluindo FFmpeg. Não promete interrupção instantânea. SIGTERM também encerra o processamento.
6. O scheduler varre até 100 leases expirados por ciclo de 10 segundos. Recoloca em queued enquanto attempts < 3; depois marca failed com LEASE_EXPIRED. Falha de motor ou timeout termina a solicitação como failed; não repete automaticamente erro determinístico.
7. A publicação bloqueia primeiro o proprietário e depois o job, revalida as referências e só conclui com token/lease vigentes. Rollback desfaz todos os registros se houver falha. Tentativa antiga ou repetição após conclusão não duplica resultados.

Códigos públicos: ENGINE_FAILED, INPUT_UNAVAILABLE, EXECUTION_TIMEOUT e LEASE_EXPIRED. Não são publicados traceback, token interno, credencial ou caminho de armazenamento. Um erro que impede gravar a falha deixa o job para recuperação após expirar a posse.

O texto/TXT, os segmentos e os anexos são publicados junto da mensagem `assistant/result`; seu `_id` aparece em output_message_id. Em erro, o cliente consulta processing_state/error_code; esta versão não cria mensagem assistant/error. Para repetir solicitação finalizada, use nova Idempotency-Key.

## Resultados e autorização

| Rota /api/v1 | Permissão |
| --- | --- |
| GET /jobs/{job_id}/outputs | dubbing.read |
| GET /jobs/{job_id}/outputs/{output_id}/file | dubbing.download |
| POST /jobs/{job_id}/cancel | dubbing.cancel |

Todas exigem Bearer e proprietário da conversa no escritório do actor. Conhecer um `_id` ou ter cargo admin não autoriza conversa alheia. Download revalida estados e usa `Cache-Control: no-store`. Listagem devolve unique_id, format, purpose, size_bytes e duration_ms; nunca storage_key. Jobs ainda sem resultado devolvem lista vazia. O histórico de mensagens inclui a resposta assistant/result.

## Docker e modelos locais

A imagem da API continua leve. `Dockerfile.WORKER.DEV` instala PyTorch CPU 2.5.1, torchaudio 2.5.1, TTS 0.22.0, Whisper 20240930, transformers 4.45.2 e FFmpeg. `requirements-worker.lock` registra as versões resolvidas na imagem; o build executa pip check. setuptools fica abaixo de 81 por compatibilidade de construção do Whisper legado.

Volumes: mysql_data para MySQL, media_data privado compartilhado pela API/worker e model_data para checkpoints. O worker monta model_data somente para leitura. Worker e scheduler não publicam portas e usam somente a rede local; apenas a API atende à rede SiPlug.

Primeira instalação ou atualização:

```sh
docker compose build app worker scheduler
docker compose up -d --wait db
docker compose run --rm --no-deps app python -m alembic upgrade head
docker compose up -d --wait db app
```

Antes de iniciar o worker, importar o cache existente **uma vez**. Não há download automático, aceitação automática de licença nem importação dos cadastros do legado. Os arquivos esperados são config.json, model.pth, vocab.json, speakers_xtts.pth e tos_agreed.txt em XTTS; tiny.pt em Whisper. Ajuste os caminhos abaixo ao host:

```sh
docker compose run --rm --no-deps \
  -v "siplug-dubber_model_data:/var/cache/dubber/models:rw" \
  -v "$HOME/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2:/import/xtts_v2:ro" \
  -v "$HOME/.cache/whisper:/import/whisper:ro" \
  worker python -m app.console.import_audio_models

docker compose up -d --wait worker scheduler
docker compose ps
```

O prefixo do volume acompanha o nome do projeto Compose; ajuste se usar `-p`. O importador recusa cache já inicializado: troca de modelos exige outro volume e nova validação. O manifest SHA-256 é verificado no preflight quando presente. Não carregar checkpoints enviados pela API; amostras de voz são apenas WAV.

O preflight carrega ambos os motores antes de começar a consumir a fila. Falha no cache impede a inicialização do worker; a API permanece independente. O worker carrega os modelos novamente em cada subprocesso: isolamento e cancelamento custam tempo de carga. Nesta versão, uma execução por worker e duas threads de CPU; dimensionar memória antes de escalar réplicas. O limite de 20 minutos pode ser ajustado por DUBBER_WORKER_TIMEOUT_SECONDS no .env.

## Limites e manutenção

WAV de saída: mono PCM16/24 kHz. TXT: UTF-8. Ajustes de velocidade e pitch usam FFmpeg. preserve_timing sintetiza cada segmento reconhecido, ajusta sua duração e alinha ao áudio de origem, com silêncio entre trechos; não é garantia de sincronização labial. O idioma selecionado em speech_to_speech orienta o reconhecimento/síntese; não há tradução entre idiomas. Whisper tiny prioriza custo local; qualidade precisa ser avaliada em amostras reais.

Cada tentativa tem diretório privado temporário. Encerramento normal, erro ou cancelamento limpa esse diretório. SIGKILL do contêiner pode deixar diretórios temporários; falha após salvar arquivo e antes de publicar pode deixar arquivo órfão privado. O scheduler também remove órfãos antigos sob bloqueio de publicação, conforme CATALOGO_PROJETOS_E_MIDIA.md; nunca remover media_data ou model_data como rotina de limpeza.

Edição de transcrições, entrada MP3/MP4, tags e vínculos externos de projetos foram acrescentados pela revisão 0005. Tradução, UI e remontagem/exportação de vídeo não fazem parte deste serviço implementado.

## Validação

Suíte padrão somente em banco descartável:

```sh
docker compose -p siplug-dubber-worker-check -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from app
docker compose -p siplug-dubber-worker-check -f docker-compose.test.yml down -v
```

`tests/processing_scenarios.py` cobre claims concorrentes, heartbeat, rejeição de token antigo, recuperação até limite, rollback de publicação, publicação única, cancelamento em execução e downloads entre proprietários/escritórios. `test_execution_supervisor.py` cobre cancelamento, deadline, encerramento do grupo de processos e ausência de senha no ambiente do filho. Fixtures simuladas são exclusivas dos testes.

`tests/real_audio_smoke.py` é uma verificação manual com os motores reais: exige DUBBER_INTEGRATION=1, banco auth_test vazio, cache e amostra locais. Valida upgrade 0003 → 0004 preservando credencial, API → caso de uso do worker → motor → banco → download autenticado para as três operações. Não executá-lo no banco da aplicação.

Referências dos motores: [XTTS](https://docs.coqui.ai/en/latest/models/xtts.html) e [Whisper](https://github.com/openai/whisper).
