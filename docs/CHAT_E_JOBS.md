# Chat, mensagens e jobs — SiPlug Dubber

> Etapa de processamento: revisão 0004_processing_results implementa resultados, transcrições e segmentos (23 tabelas de domínio). Worker CPU com XTTS/Whisper, scheduler de leases, cancelamento em execução e download privado disponíveis. Consulte WORKER_E_PROCESSAMENTO.md para o estado atual e comandos; os registros de etapas anteriores abaixo são históricos.

## Estado implementado

A revisão `0003_chat_jobs` acrescenta dubbing_chats, dubbing_messages, dubbing_message_files e dubbing_jobs. O banco passa a ter 20 tabelas de domínio, além da tabela de controle Alembic. As referências são VARCHAR(255) para `_id`, com FKs compostas por escritório; reply_to_message_id é limitado à mesma conversa. Status continua active/inactive, distinto de processing_state.

A submissão retorna HTTP 202 com processing_state=queued. O worker implementado na revisão 0004_processing_results consome a fila, executa XTTS/Whisper e publica resultados e resposta assistant/result. Consulte WORKER_E_PROCESSAMENTO.md para instalação do cache, execução, recuperação e downloads.

Tags, voice_tags e project_audio_links ainda não foram implementadas. Resultados, transcrições e segmentos foram acrescentados pela revisão 0004_processing_results, totalizando 23 tabelas de domínio. Não há importação automática do legado.

## Responsabilidade dos arquivos

| Local | Responsabilidade |
| --- | --- |
| routes/chat.py | Declarar as rotas, incluídas no router autenticado |
| http/controllers/chat/*_controller.py | Um controller por operação; HTTP → DTO → caso de uso → resposta |
| http/schemas/*_input.py | Schema próprio para forma e limites do JSON |
| use_cases/*/dtos/*_dto_in.py e *_dto_out.py | Contratos internos separados e independentes de FastAPI/SQLAlchemy |
| use_cases/*/*_use_case_service.py | Orquestrar serviços e a transação; sem SQL |
| entities/dubbing_chat, dubbing_message e dubbing_job | Entidades imutáveis, contratos de repositório e serviços pequenos com DTOs próprios |
| services/chat | Autorização de recursos, preparo de áudio, canonicalização da solicitação e representação pública do job |
| interfaces/i_chat_unit_of_work.py | Contrato da transação com os repositórios necessários |
| models/chat/sqlalchemy_chat_unit_of_work.py | Abrir uma conexão/transação compartilhada, commit/rollback e fechamento |
| models/chat/*_repository.py | SQL parametrizado e conversão para entidades; não resolve actor nem decide permissão HTTP |
| providers/container.py | Injetar fábrica de Unit of Work e storage; nenhum singleton de conexão/transação |

O Unit of Work é novo por operação. Sem commit explícito, ocorre rollback. O caso de uso não conhece Connection, Session ou Model. Arquivo, entidade, DTO e controller não carregam motores de IA.

## Autorização e propriedade

Toda rota exige Bearer. O escritório e o proprietário vêm do actor, nunca do JSON. O proprietário é user_customers._id, não users.id. Ter a permissão correta não permite ler conversa de outro usuário do mesmo escritório. Nesta etapa não há acesso administrativo a conversas alheias; um fluxo assim exigirá permissão específica e implementação própria.

A migration concede sete permissões explicitamente aos cargos admin ativos que já têm user.update ativo do escritório. Novos bootstraps também recebem essas permissões. Não existe bypass de administrador em runtime. Credenciais existentes são preservadas.

## Rotas (prefixo /api/v1)

| Método e rota | Permissão | Resultado |
| --- | --- | --- |
| POST /chats | dubbing_chat.create | Cria conversa do actor, 201 |
| GET /chats | dubbing_chat.read | Lista somente conversas próprias, inclusive inativas |
| GET /chats/{chat_id} | dubbing_chat.read | Metadados da própria conversa |
| PUT /chats/{chat_id} | dubbing_chat.update | Substitui título, seleção de voz e status |
| GET /chats/{chat_id}/messages | dubbing_chat.read | Mensagens ativas em ordem de criação |
| POST /chats/{chat_id}/audio | dubbing.generate | Recebe WAV e cria mensagem user/audio, 201 |
| GET /chats/{chat_id}/messages/{message_id}/audio | dubbing.download | Download privado do áudio de entrada |
| POST /chats/{chat_id}/jobs | dubbing.generate | Aceita solicitação ou devolve repetição idempotente, 202 |
| GET /chats/{chat_id}/jobs | dubbing.read | Lista jobs próprios da conversa, mais recentes primeiro |
| GET /jobs/{job_id} | dubbing.read | Estado e metadados seguros do job |
| POST /jobs/{job_id}/cancel | dubbing.cancel | Cancela job queued/processing; repetir cancelamento é permitido |

Listagens aceitam limit (1–100, padrão 20) e offset (mínimo 0). Conversa inativa bloqueia mensagens, áudio e novas solicitações. Não é possível inativar conversa com job queued/processing: retorna 409; cancele os jobs elegíveis primeiro. PUT é substituição de metadados: campos opcionais omitidos ficam NULL e status omitido assume active. Não muda snapshots de jobs anteriores.

Erros: 401 sem autenticação; 403 sem permissão; 404 recurso de outro escritório/proprietário ou indisponível; 409 conflito de idempotência/estado; 422 contrato/referência inválidos. As respostas não expõem worker_token, request_hash ou storage_key.

## Criar conversa

```http
POST /api/v1/chats
Authorization: Bearer SEU_TOKEN
Content-Type: application/json
```

```json
{"title":"Apresentação","selected_voice_id":"ID_DA_VOZ"}
```

Voz selecionada deve ser ativa, do escritório e ter amostra tecnicamente pronta com arquivo disponível. selected_voice_id pode ser NULL para criar uma conversa sem voz; transcribe não exige voz. IDs retornam em data.unique_id. O título pode ser NULL.

## Enviar texto e criar job

```http
POST /api/v1/chats/ID_DO_CHAT/jobs
Authorization: Bearer SEU_TOKEN
Idempotency-Key: apresentacao-001
Content-Type: application/json
```

```json
{
  "operation":"text_to_speech",
  "input_text":"Olá! Este é o texto da apresentação.",
  "speed":1.00,
  "pitch_semitones":0.00
}
```

A mensagem user/text e o job são criados na mesma transação. O texto deve ter de 1 a 10000 caracteres após remover espaços externos. Autor e role são definidos pelo servidor. Não há edição ou exclusão de mensagens já aceitas; envie uma nova solicitação para outro conteúdo.

O job registra input_text, voice_sample_id efetivo, idioma alvo e parâmetros. A seleção posterior de outra voz ou amostra não modifica o job. target_language_id é opcional no pedido: em geração, usa o idioma da voz quando omitido, validando catálogo ativo. A compatibilidade com o idioma do motor é validada na execução; cadastrar idioma não equivale a implementar um motor para ele.

## Enviar áudio e criar job

Primeiro envie o áudio binário, sem multipart:

```sh
curl -X POST 'http://localhost:8087/api/v1/chats/ID_DO_CHAT/audio' \
  -H "Authorization: Bearer $DUBBER_TOKEN" \
  -H 'Content-Type: audio/wav' \
  --data-binary @entrada.wav
```

WAV PCM mono de 16 bits, 16000/22050/24000/44100/48000 Hz, entre 3 e 180 segundos, até 20 MiB. Conteúdo truncado ou totalmente zerado é rejeitado. 415 para tipo incompatível; 413 para tamanho excedido; 422 para WAV inválido. Os limites atuais reutilizam a validação técnica das amostras; conversão de MP3/vídeo e áudios longos serão etapas próprias.

Guarde data.unique_id da mensagem retornada. Upload repetido cria outra mensagem; a idempotência descrita abaixo se aplica à submissão do job, não ao upload.

Depois envie POST /chats/{chat_id}/jobs com outra Idempotency-Key:

```json
{
  "operation":"speech_to_speech",
  "input_message_id":"ID_DA_MENSAGEM_AUDIO",
  "preserve_timing":true,
  "speed":1.00,
  "pitch_semitones":0.00
}
```

A mensagem deve pertencer à mesma conversa e ao mesmo proprietário, ser user/audio ativa e ter anexo source disponível. O cliente não passa caminho, storage_key, media_file_id ou role. O job captura o arquivo recebido e a amostra da voz selecionada. preserve_timing alinha a síntese aos trechos reconhecidos e à duração da origem; consulte os limites de qualidade na documentação do worker.

Para solicitar apenas transcrição:

```json
{"operation":"transcribe","input_message_id":"ID_DA_MENSAGEM_AUDIO"}
```

Transcrição dispensa voz e permite target_language_id NULL para detecção pelo Whisper. Não aceita texto junto, ajuste de velocidade/pitch nem preserve_timing. Em conversão, velocidade varia de 0.50 a 1.50 e pitch de -6.00 a 6.00, com no máximo duas casas decimais; preserve_timing só vale para speech_to_speech.

## Idempotência e estados

Idempotency-Key é obrigatória, com 1–100 caracteres ASCII entre letras, dígitos, ponto, hífen, sublinhado e dois-pontos; comparação sensível a caixa. O escopo é (office_id, user_customer_id, idempotency_key), inclusive entre conversas. Request_hash usa SHA-256 da solicitação canonicalizada: texto sem espaços externos, decimais com duas casas, operação, chat e referências/parâmetros solicitados. Não inclui credencial.

Mesma chave e solicitação igual devolvem o mesmo job sem duplicar mensagem. Chave já usada com outro conteúdo/conversa retorna 409. Isso também é verificado quando chamadas chegam simultaneamente. A seleção de voz do chat não é usada para recalcular o hash da repetição: o job anterior conserva sua amostra. Job cancelado continua associado à chave; para nova execução, use nova chave.

O cancelamento altera processing_state para cancelled e finished_at, sem usar status=inactive. O estado processing também é cancelável: invalida a posse e o supervisor encerra a inferência no próximo heartbeat. Estados completed/failed retornam 409.

## Transação, arquivos e próximos passos

As mutações seguem uma ordem de bloqueios por proprietário antes do chat. Mensagem e job compartilham a transação; falha de persistência desfaz ambas. Não há inferência ou processamento de modelo dentro dela. O banco é a fila durável nesta etapa, sem broker ou worker fictício.

O áudio usa o volume privado existente media_data. Arquivos e banco não compartilham transação: falha após gravação pode deixar arquivo órfão privado, nunca publicado sem registro autorizado; a reconciliação automática de arquivos ainda está pendente; o scheduler atual recupera somente jobs. O endpoint de download sempre revalida proprietário, estado e caminho privado, com Cache-Control no-store.

Worker, motores reais, recuperação, transcrições e respostas assistant estão implementados. A instalação do cache e o fluxo de resultados são documentados em WORKER_E_PROCESSAMENTO.md.

## Aplicar e validar

```sh
docker compose build app
docker compose run --rm --no-deps app python -m alembic upgrade head
docker compose up -d --wait
```

Teste somente com o Compose efêmero:

```sh
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from app
docker compose -f docker-compose.test.yml down -v
```

A suíte cobre contratos, autenticação, proprietário/empresa, texto/áudio, idempotência concorrente, rollback, snapshots, histórico e cancelamento. Downgrade é destrutivo e só é validado em banco descartável.
