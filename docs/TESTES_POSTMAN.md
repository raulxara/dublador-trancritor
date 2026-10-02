# Testar o Dubber pelo Postman

Este roteiro valida os principais casos de uso pela API real, incluindo processamento no worker. A coleção contém 52 requisições, com IDs encadeados e verificações automáticas. Ela cria registros reais de teste: voz, amostra, tag, chat, mensagens, jobs e resultados. O vínculo de projeto é removido logicamente durante o teste; os demais registros permanecem para inspeção.

## 1. Conferir o ambiente

No terminal:

```sh
cd /Users/usuario/Desktop/work/ProjetoPessoal/Projects/dubber
docker compose ps
```

Confira `app`, `db`, `worker` e `scheduler`. Modelos e migrations devem estar instalados conforme [Worker e processamento](WORKER_E_PROCESSAMENTO.md). Health e readiness HTTP não comprovam que o worker consegue executar os motores de áudio.

## 2. Importar e configurar

No Postman desktop, use **Import** para importar:

- [Coleção SiPlug Dubber](postman/SiPlug_Dubber.postman_collection.json)
- [Ambiente local](postman/SiPlug_Dubber_Local.postman_environment.json)

Selecione o ambiente importado e preencha `token` com o token original emitido ao criar o administrador, sem escrever `Bearer`. Mantenha esse valor apenas no seu ambiente local. O arquivo distribuído está sem credenciais; não publique uma exportação com seu token preenchido.

`base_url` começa com `http://127.0.0.1:8087`, sem barra final. Ajuste se a porta local foi alterada. O Postman acrescenta `Authorization: Bearer {{token}}` às requisições protegidas. O token não deve ir no JSON, na query string ou em um campo de usuário. Senha do banco e hash armazenado não servem como token da API. Consulte [Autenticação](AUTENTICACAO.md) se precisar conferir o provisionamento; este roteiro não rotaciona nem revoga o acesso existente.

## 3. Selecionar os arquivos

Nas requisições **02.2** e **05.1**, abra **Body → binary → Select File**, selecione um WAV local e salve a requisição. Não use `form-data`.

Use fala audível, mono PCM16, entre 3 e 180 segundos, até 20 MiB; taxas aceitas: 16000, 22050, 24000, 44100 e 48000 Hz. Na amostra de voz, prefira uma única pessoa falando, sem música ou ruído. É possível selecionar o mesmo WAV nas duas requisições. Há uma amostra preservada em `data/voices/caf91344/clean.wav`, caso queira usá-la no teste local.

Os caminhos dos arquivos ficam a cargo do seu Postman. Execute localmente; uma execução na nuvem não recebe automaticamente os arquivos do seu computador.

## 4. Executar o fluxo principal

Na primeira execução, use **Send** em ordem, das pastas **00 a 09**, acompanhando o corpo da resposta e a aba **Test Results**.

| Pasta | O que conferir |
| --- | --- |
| 00 — Saúde | Health e readiness retornam 200. |
| 01 — Contexto e catálogos | Token aceito, permissões disponíveis e idioma pt-BR selecionado. |
| 02 — Voz e amostra | Voz criada; amostra pronta e vinculada; download do WAV. |
| 03 — Tags | Tag criada e associada à voz. |
| 04 — Texto para voz | Chat criado, job concluído, áudio disponível e repetição sem duplicar o job. |
| 05 — Áudio para transcrição | Upload, transcrição concluída e TXT disponível. |
| 06 — Editar transcrição | Nova versão criada; base antiga rejeitada; histórico preservado. |
| 07 — Áudio para outra voz | Reconhecimento e síntese usando a voz selecionada, com WAV e TXT. |
| 08 — Vínculo de projeto | Vínculo criado sem duplicação e removido sem apagar o áudio. |
| 09 — Rejeições esperadas | Sem token e token inválido: 401; office_id no JSON: 422; conflitos: 409. |

A requisição **01.1** inicia uma execução nova: limpa os IDs da coleção e gera um `run_id`. Os IDs seguintes são salvos automaticamente. Não a repita no meio do fluxo, pois isso perde as referências da execução em andamento. Uma execução nova cria novos registros.

Nas consultas **04.4**, **05.4** e **07.2**, aguarde `processing_state: completed` antes de continuar. Ao usar Send, repita a consulta enquanto estiver `queued` ou `processing`. Um POST com 202 significa que o pedido foi aceito, não que o áudio já está pronto. Se aparecer `failed`, examine `error_code` e os logs antes de prosseguir.

Nos downloads, use a opção de salvar a resposta em arquivo. Ouça o WAV e confira inteligibilidade, voz selecionada, pausas e duração. Leia o TXT e compare com a fala original. Um HTTP 200 não comprova qualidade do áudio ou precisão da transcrição.

O identificador de projeto usado é fictício (`postman-{{run_id}}`). Essa etapa testa o vínculo local; o backend da SiPlug continua responsável por verificar existência e autorização do projeto externo.

## 5. Executar automaticamente no Collection Runner

Depois de selecionar e salvar os arquivos, execute a coleção localmente pelo **Runner**, com uma iteração e a ordem original. As consultas de job repetem automaticamente com intervalo aproximado de dois segundos. O limite padrão por job é 25 minutos (`poll_timeout_ms = 1500000`); o script aceita entre 1 e 30 minutos. O processamento em CPU pode demorar.

Respostas HTTP inesperadas interrompem a sequência. Confira também todas as asserções em Test Results: algumas verificações de conteúdo podem falhar mesmo com HTTP correto. Depois de corrigir um problema, prefira reiniciar uma execução completa; repetir isoladamente criação de tag ou edição de transcrição pode gerar conflito esperado pelo estado já alterado.

## 6. Testes opcionais

As pastas 10 a 12 são puladas por padrão. Habilite somente após concluir o fluxo principal:

| Variável do ambiente | Preparação e resultado esperado |
| --- | --- |
| `run_media_tests = true` | Selecione MP3 na 10.1 e MP4 com faixa de áudio na 10.2, em Body binary. Confere aceitação e download da entrada normalizada em WAV. |
| `run_scope_tests = true` | Preencha `other_token` com credencial de outro usuário/proprietário, com permissões `dubbing.read` e `dubbing.download`. Espera 404 ao consultar ou baixar o job do primeiro usuário. Usar o mesmo token invalida o cenário; 403 prova falta de permissão, não isolamento de propriedade. |
| `run_cancel_tests = true` | Cria outro job e o cancela imediatamente; espera estado cancelled. Se ele terminar antes do cancelamento, 409 é uma disputa legítima de execução; repita em uma execução nova. |

Essas variáveis usam o texto `true` ou `false`. O fluxo de MP3/MP4 valida entrada; não há exportação MP3 nem geração de vídeo dublado. A coleção não cria automaticamente uma segunda identidade.

## 7. Diagnóstico

| Resultado | Interpretação |
| --- | --- |
| 200 / 201 | Consulta/alteração concluída ou recurso criado. |
| 202 | Job aceito; consultar até concluir. |
| 401 | Credencial ausente, inválida ou sem validade. |
| 403 | Identidade autenticada sem a permissão exigida. |
| 404 | Recurso inexistente ou fora do escopo permitido. |
| 409 | Conflito de estado, idempotência ou versão. |
| 415 / 422 | Tipo de conteúdo, arquivo ou campos incompatíveis com o contrato. |
| 503 | Dependência indisponível; conferir serviços e resposta. |

Para investigar jobs parados ou com falha:

```sh
docker compose logs --tail=100 app worker scheduler
```

Limpeza automática de órfãos é uma rotina interna do scheduler, sem endpoint neste roteiro. Seu comportamento e critérios estão em [Catálogo, projetos e mídia](CATALOGO_PROJETOS_E_MIDIA.md). Não remova arquivos ou volumes para simular essa rotina durante o teste principal.

## Escopo da validação

Os arquivos JSON e a sintaxe dos 98 scripts foram validados na preparação desta coleção. A execução completa desta coleção com sua credencial e seus arquivos deve ser feita no Postman; não foi executada como parte da entrega destes arquivos. A aprovação do roteiro fornece evidências dos cenários exercitados, sem substituir avaliação de segurança e configuração do ambiente de publicação.

Referências do Postman: [importação](https://learning.postman.com/docs/getting-started/importing-and-exporting/importing-data), [controle de execução e skipRequest](https://learning.postman.com/docs/tests-and-scripts/write-scripts/postman-sandbox-reference/pm-execution/) e [arquivos de teste locais](https://learning.postman.com/docs/use/send-requests/create-requests/test-data).
