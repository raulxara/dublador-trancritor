# Vozes e amostras — API SiPlug Dubber

## Implementado

A revisão `0002_voices` acrescenta languages, gender, voices, media_files e voice_samples. São 16 tabelas de domínio implementadas, além da tabela de controle Alembic; tags, chat e processamento serão acrescentados posteriormente. Toda FK referencia `_id` VARCHAR(255). FKs compostas impedem associações entre escritórios e amostra atual de outra voz. Status é active/inactive, separado de validation_state/storage_state.

A migration provisiona idiomas pt-BR, en, es e classificações neutra, feminino, masculino. São catálogos iniciais; o idioma cadastrado não garante suporte de um motor. IDs são UUIDs determinísticos gerados no servidor. Liste o catálogo para obter os IDs aceitos.

A migration concede explicitamente voice.read/register/update aos cargos admin ativos que já possuíam user.update ativo no próprio escritório. Não existe bypass em runtime. Novos bootstraps também recebem essas permissões. Usuário, senha e token existentes não são substituídos. Outros cargos precisam de concessões explícitas futuras.

## Rotas

Todas exigem Authorization: Bearer e utilizam exclusivamente o escritório do actor do servidor.

| Método e rota /api/v1 | Permissão | Comportamento |
| --- | --- | --- |
| GET /languages | catalog.read | Idiomas ativos |
| GET /gender | catalog.read | Classificações ativas |
| POST /voices | voice.register | Cadastra voz ativa |
| GET /voices?limit=20&offset=0 | voice.read | Lista vozes do escritório, inclusive inativas; máximo 100 por página |
| GET /voices/{voice_id} | voice.read | Consulta voz do escritório |
| PUT /voices/{voice_id} | voice.update | Substitui metadados e status; não apaga áudios |
| POST /voices/{voice_id}/samples | voice.update | Envia WAV, cria nova versão e seleciona como amostra atual |
| GET /voices/{voice_id}/samples?limit=20&offset=0 | voice.read | Lista versões da voz ativa, máximo 100 por página |
| GET /voices/{voice_id}/samples/{sample_id}/audio | voice.read | Download privado; exige voz/amostra/arquivo ativos e prontos |

Exemplo de corpo para cadastro:

```json
{
  "name": "Voz de apresentação",
  "language_id": "ID_RETORNADO_EM_LANGUAGES",
  "gender_id": null,
  "description": "Voz para vídeos institucionais"
}
```

Em PUT, envie os mesmos campos e `status` como active ou inactive. PUT substitui os metadados, não é atualização parcial; status omitido assume active. Campos extras, como office_id, permissions ou current_sample_id, são rejeitados. O identificador da voz aparece em `data.unique_id`.

## Enviar amostra

Envie o arquivo no corpo binário da requisição, sem JSON ou multipart:

```sh
curl -X POST "http://localhost:8087/api/v1/voices/ID_DA_VOZ/samples" \
  -H "Authorization: Bearer $DUBBER_TOKEN" \
  -H "Content-Type: audio/wav" \
  --data-binary @amostra.wav
```

DUBBER_TOKEN representa o token configurado no seu backend/ambiente local, não uma chave incorporada no frontend.

Aceita WAV PCM, mono, 16 bits, taxas 16000/22050/24000/44100/48000 Hz, de 3 a 180 segundos, até 20 MiB. O servidor valida cabeçalho, frames reais, duração, tamanho e rejeita áudio totalmente zerado. Content-Type incompatível retorna 415; tamanho acima do limite retorna 413; conteúdo inválido retorna 422. Arquivo truncado é rejeitado.

`validation_state=ready` nesta etapa significa validação técnica do WAV; não certifica qualidade vocal, consentimento, idioma nem compatibilidade com um modelo específico. A análise de ruído/clipping, conversão de outros formatos e normalização pelo worker ainda não foram integradas. `normalized_file_id` permanece NULL. O nome original armazenado é sample.wav; não é recebido como caminho do cliente.

Cada envio válido cria uma amostra imutável, com UUID e versão crescente. O lock da voz serializa uploads concorrentes e a transação registra arquivo, amostra e seleção atual. A versão anterior permanece preservada. Inativar a voz bloqueia novos envios e downloads; não apaga arquivos. Não há exclusão física nem seleção arbitrária de amostra antiga nesta etapa.

## Armazenamento

Bytes ficam no volume Docker `siplug-dubber_media_data`, montado em `/var/lib/dubber/media`, separado do banco. As chaves são geradas pelo servidor com namespace derivado do escritório e UUID do arquivo; não são retornadas pela API. Download autenticado usa Cache-Control no-store. A pasta não é publicada como conteúdo estático.

Faça backup do volume de arquivos junto com o MySQL. `docker compose down` preserva ambos. Um erro de banco após gravação pode deixar arquivo privado órfão; a limpeza reconciliada será implementada com os jobs. Não removemos o arquivo em erro de commit ambíguo para evitar apagar bytes de um registro possivelmente confirmado.

Os áudios legados em data/ continuam intactos e não foram importados automaticamente.

## Arquitetura e execução

Controller valida HTTP e transforma DTO; use case orquestra a operação; service aplica a regra; interface define persistência; repositório SQLAlchemy executa consultas parametrizadas com office_id. Storage e validação WAV são serviços separados. Não há SQL nos controllers ou use cases.

```sh
docker compose build app
docker compose run --rm --no-deps app python -m alembic upgrade head
docker compose up -d --wait
```

Os testes usam o Compose isolado, nunca o banco com seu administrador:

```sh
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from app
docker compose -f docker-compose.test.yml down -v
```

Cobertura: autenticação e permissões, isolamento entre escritórios, referências inválidas, inativação, conteúdo WAV/truncamento, download privado, FKs de amostra atual e versionamento com uploads concorrentes.

Próxima etapa: tags, chat, mensagens e jobs de dublagem; depois integração dos motores/worker para transformar texto ou áudio com a voz selecionada.
