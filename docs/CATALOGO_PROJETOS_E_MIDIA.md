# Catálogo, projetos, transcrições e mídia — SiPlug Dubber

## Estado implementado em 11/09/2026

A revisão `0005_catalog_and_projects` completa as **26 tabelas de domínio** do modelo: tags, voice_tags e project_audio_links foram acrescentadas às 23 anteriores. A tabela de controle Alembic é adicional. IDs técnicos continuam INT; `_id` e todas as referências `*_id` são VARCHAR(255). Relações privadas usam FKs compostas com office_id, status active/inactive e bloqueio de exclusão física. external_project_id é uma referência externa, sem FK entre aplicações.

Também foram implementadas edição versionada de transcrições, entrada de MP3/MP4 no chat e reconciliação automática de arquivos órfãos. O frontend continua na SiPlug. Este documento descreve o comportamento implementado; os documentos de banco/arquitetura conservam o desenho completo e registros históricos.

## Responsabilidade dos arquivos

Os caminhos abaixo são relativos a src/app/. Cada operação tem controller, caso de uso, serviço e DTOs separados; nenhuma rota executa SQL ou usa token recebido no JSON.

| Local | Responsabilidade |
| --- | --- |
| entities/tag, project_audio_link, transcription/*_entity.py | Entidades imutáveis; identificadores e dados de domínio |
| entities/*/i_*_repository.py | Contratos de persistência sem SQLAlchemy/FastAPI |
| entities/tag/services/create_tag e update_tag | Validar nome/slug/status e persistir a entidade |
| entities/tag/services/list_tags e list_voice_tags | Consultar catálogo e associações ativas |
| entities/tag/services/set_voice_tags | Validar voz/tags do escritório e substituir o conjunto, até 50 tags |
| entities/project_audio_link/services/create_project_audio_link | Validar resultado de áudio próprio e criar/reativar vínculo idempotente |
| entities/project_audio_link/services/list_project_audio_links e unlink_project_audio | Listar vínculos próprios e inativar sem apagar áudio |
| entities/transcription/services/list_transcriptions e edit_transcription | Histórico e criação de nova versão com controle de versão-base |
| */dtos/*_dto_in.py e *_dto_out.py | Contratos próprios de entrada/saída, em arquivos separados |
| use_cases/*/*_use_case_service.py | Permissão do actor, propriedade, serviços e transação |
| models/catalog/*_repository.py | SQL parametrizado, bloqueios e conversão de registros para entidades |
| models/chat/sqlalchemy_chat_unit_of_work.py | Uma conexão/transação por operação, incluindo os novos repositórios |
| http/schemas/*_input.py | Formato e limites HTTP; rejeita campos desconhecidos |
| http/controllers/catalog/*_controller.py | HTTP → DTO → caso de uso → resposta |
| routes/catalog.py | Declarar as rotas, incluídas no router autenticado |
| services/media/normalize_media_service.py | Converter bytes locais MP3/MP4 em WAV com FFmpeg e limites de tempo/duração |
| services/media/media_publication_guard.py | Bloqueio de filesystem compartilhado entre publicação e limpeza |
| services/media/orphan_cleanup_service.py | Examinar candidatos antigos e remover somente órfãos reconhecidos |
| models/media/sqlalchemy_media_inventory_repository.py | Consultar existência de qualquer registro media_files pela chave |
| use_cases/cleanup_orphan_media | Orquestrar reconciliação pelo contrato, sem acesso ao filesystem |
| providers/container.py e scheduler_container.py | Compor dependências HTTP e de manutenção |
| scheduler/main.py | Recuperar jobs e executar ciclos limitados de reconciliação |

## Permissões e escopo

Novas permissões: tag.read, tag.update, project_audio.read, project_audio.update e transcription.update. A migration concede-as explicitamente aos cargos admin ativos com user.update ativo. O bootstrap concede-as a novos administradores. Não há bypass administrativo em runtime. As credenciais existentes não são recriadas.

Tags são compartilhadas dentro do escritório; gerenciamento exige tag.update. Associações de voz usam voice.read/voice.update. Transcrições e vínculos de projetos são privados ao proprietário do áudio/conversa, mesmo para outro usuário do mesmo escritório com as mesmas permissões. office_id, user_id e owner_id vêm exclusivamente do actor resolvido pelo Bearer.

Erros: 401 sem autenticação, 403 sem permissão, 404 recurso indisponível/de outro escopo, 409 conflito de versão/slug ou saturação temporária do conversor, 422 conteúdo inválido.

## Tags e associação com vozes

| Método e rota /api/v1 | Permissão | Comportamento |
| --- | --- | --- |
| POST /tags | tag.update | Cria tag, 201 |
| GET /tags | tag.read | Lista tags do escritório, inclusive inativas |
| PUT /tags/{tag_id} | tag.update | Substitui nome, slug e status |
| GET /voices/{voice_id}/tags | voice.read | Lista associações e tags ativas de voz ativa |
| PUT /voices/{voice_id}/tags | voice.update | Substitui todo o conjunto de tags da voz |

POST /tags recebe `{"name":"Narrador","slug":"narrador"}`. PUT recebe também `"status":"active"` ou `"inactive"`. Nome tem 1–255 caracteres após trim; slug usa letras ASCII minúsculas, números e hífens internos, até 255 caracteres. O slug é único por escritório; colisão retorna 409.

PUT de associações recebe `{"tag_ids":["ID_DA_TAG"]}`. Até 50 IDs únicos. Lista vazia remove logicamente todas as associações. Voz e tags precisam estar ativas e no mesmo escritório; qualquer referência inválida desfaz toda a alteração. As associações anteriores são inativadas e as selecionadas criadas/reativadas. Inativar uma tag a oculta das associações consultadas; reativá-la pode voltar a exibir associações ainda ativas.

## Vínculos com projetos da SiPlug

| Método e rota /api/v1 | Permissão | Comportamento |
| --- | --- | --- |
| POST /projects/{project_id}/audio-links | project_audio.update | Vincula resultado de áudio próprio, 201 |
| GET /projects/{project_id}/audio-links | project_audio.read | Lista vínculos ativos próprios do projeto |
| DELETE /project-audio-links/{link_id} | project_audio.update | Inativa vínculo próprio, 200; repetir é permitido |

POST recebe `{"output_id":"ID_DO_RESULTADO"}`. Use o unique_id retornado por GET /jobs/{job_id}/outputs com purpose=audio, não um caminho nem um media_file_id arbitrário. O serviço resolve o arquivo e verifica job concluído, proprietário, conversa e arquivos ativos. O vínculo devolve media_file_id público, nunca storage_key.

Repetir o mesmo par projeto/arquivo devolve o mesmo vínculo. Um vínculo inativo é reativado. A exclusão lógica não apaga o áudio nem o projeto. Projeto externo aceita identificador ASCII de 1–255 caracteres com letras, números, ponto, hífen, sublinhado e dois-pontos, iniciando por letra/número.

**A SiPlug deve validar existência e permissão do projeto antes de chamar o Dubber.** Este serviço não consulta nem replica o banco de projetos. A implementação é o lado Dubber do contrato; a integração na aplicação consumidora não foi alterada. Vínculos removidos externamente precisam ser inativados pela SiPlug. O download continua pela rota autenticada de resultados e revalida o acesso à conversa.

## Edição e versionamento de transcrições

| Método e rota /api/v1 | Permissão |
| --- | --- |
| GET /jobs/{job_id}/transcriptions | dubbing.read |
| POST /jobs/{job_id}/transcriptions | transcription.update |

GET retorna versões ativas, da mais recente para a mais antiga. Listagens paginadas de tags, projetos e transcrições aceitam limit 1–100 (padrão 20) e offset >= 0.

Para editar, obtenha o unique_id da versão atual e envie:

```json
{"base_transcription_id":"ID_DA_VERSAO_ATUAL","text":"Texto corrigido para a próxima dublagem."}
```

O job deve estar completed e pertencer ao actor numa conversa ativa. Texto após trim: 1–10000 caracteres. O serviço cria outro `_id`, incrementa version, marca origin=edited, registra edited_by_user_id e previous_transcription_id. A transação bloqueia proprietário/job e lê a versão atual. Duas edições da mesma base não se sobrescrevem: uma conclui e a outra recebe 409; recarregue o histórico antes de tentar novamente.

A transcrição reconhecida, os segmentos executados, a mensagem e o TXT/WAV publicados permanecem como registro da execução original. Editar não gera áudio automaticamente. Para nova síntese, envie o texto da versão escolhida em input_text de um novo job text_to_speech, com nova Idempotency-Key; esse texto passa a ser o snapshot da nova execução.

## Entrada de MP3 e vídeo MP4

POST /chats/{chat_id}/audio agora aceita bytes binários com Content-Type audio/wav, audio/x-wav, audio/mpeg ou video/mp4. O envio permanece sem multipart e sem URLs. Limite: 20 MiB e faixa de áudio entre 3 e 180 segundos. Vídeo precisa conter áudio; utiliza a primeira faixa de áudio.

```sh
curl -X POST 'http://localhost:8087/api/v1/chats/ID_DO_CHAT/audio' \
  -H "Authorization: Bearer $DUBBER_TOKEN" \
  -H 'Content-Type: video/mp4' \
  --data-binary @entrada.mp4
```

A API verifica permissão/propriedade antes da conversão e revalida antes de persistir. FFprobe tem timeout de 15 segundos e FFmpeg de 60; no máximo duas conversões simultâneas por processo da API. Com ambas ocupadas, retorna 409 para nova tentativa. O formato de entrada é forçado e protocolos são limitados a arquivos locais/pipe. Conteúdo inválido, vídeo sem áudio e mídia longa são rejeitados; a conversão não publica trechos truncados como se fossem o arquivo completo.

MP3/MP4 viram WAV PCM16 mono/24 kHz privado. O arquivo original e a imagem do vídeo não são armazenados. O download da mensagem devolve o WAV normalizado. Depois, use o ID da mensagem para transcribe ou speech_to_speech, como no fluxo anterior. Amostras de cadastro de voz continuam usando WAV validado; não foi mudado esse contrato.

**Este suporte é de entrada:** a aplicação não remonta/exporta vídeo dublado nem oferece saída MP3. Os resultados de síntese continuam WAV, e transcrições continuam TXT. Não há tradução entre idiomas nesta etapa.

## Limpeza automática de órfãos

O scheduler examina até 500 candidatos por ciclo, avançando um cursor entre ciclos de aproximadamente 10 segundos. Só remove arquivos privados de nomes reconhecidos (UUID WAV/TXT/temporário em pasta hash de escritório) e diretórios de tentativas job-* com mais de 24 horas. O índice ix_media_storage_key foi adicionado para a consulta global de inventário.

Qualquer registro em media_files protege o arquivo, inclusive registros inactive ou não prontos. Links simbólicos, arquivos fora da raiz, nomes desconhecidos, modelos e dados do legado não são candidatos. Se o banco não responde, a limpeza interrompe sem decidir que o arquivo é órfão.

O arquivo `.publication.lock` coordena processos por flock no volume Docker local: uploads de amostras, uploads do chat e processamento/publicação de resultados seguram bloqueio compartilhado. A limpeza precisa de bloqueio exclusivo não bloqueante; se existe publicação, adia o ciclo. O bloqueio abrange gravação e commit, evitando apagar um arquivo antes de seu registro no banco. Todos os escritores devem usar esse contrato; gravações manuais fora da aplicação não são coordenadas. O volume deve suportar flock; esta configuração foi desenhada para o volume local do Docker.

Não há endpoint público de limpeza. Não use docker compose down -v como manutenção: ele remove os volumes persistentes. A reconciliação não exclui registros do banco nem aplica retenção a áudios registrados.

## Aplicação e validação

```sh
docker compose build app worker scheduler
docker compose stop app worker scheduler
docker compose run --rm --no-deps app python -m alembic upgrade head
docker compose up -d --wait
docker compose ps
```

Na primeira instalação, inicialize db e o cache dos modelos conforme WORKER_E_PROCESSAMENTO.md. Na atualização, mantenha .env, media_data, mysql_data e model_data. Parar escritores antes da primeira subida do novo scheduler garante que todos já usem o bloqueio de publicação.

Testes somente no ambiente descartável:

```sh
docker compose -p siplug-dubber-completion-check -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from app
docker compose -p siplug-dubber-completion-check -f docker-compose.test.yml down -v
```

completion_scenarios.py cobre catálogo, rollback de associações, versionamento concorrente, vínculos idempotentes e acesso entre escritórios/proprietários. test_media_completion.py usa FFmpeg real para MP3/MP4, rejeita mídia inválida e verifica limpeza, falha do banco e bloqueio entre processos. Os testes anteriores de autenticação, worker e resultados continuam na suíte. Não são criados usuários de teste no banco da aplicação.
