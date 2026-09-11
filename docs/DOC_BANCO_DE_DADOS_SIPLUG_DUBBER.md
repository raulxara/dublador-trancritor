# SiPlug Dubber — Documentação do banco de dados

> Etapa de processamento: revisão 0004_processing_results implementa resultados, transcrições e segmentos (23 tabelas de domínio). Worker CPU com XTTS/Whisper, scheduler de leases, cancelamento em execução e download privado disponíveis. Consulte WORKER_E_PROCESSAMENTO.md para o estado atual e comandos; os registros de etapas anteriores abaixo são históricos.

> Atualização 10/09/2026: revisão 0003_chat_jobs acrescenta quatro tabelas de chat/mensagens/jobs (20 tabelas de domínio implementadas). Entrada de texto/áudio, propriedade, idempotência e cancelamento disponíveis. Jobs permanecem queued até a integração do worker. Consulte CHAT_E_JOBS.md; o restante deste documento contém especificações futuras e registros das etapas anteriores.

> Etapa de vozes implementada em 10/09/2026: revisão 0002_voices acrescenta languages, gender, voices, media_files e voice_samples, totalizando 16 tabelas de domínio. Cadastro de vozes e upload/download privado de WAV disponíveis; tags/chat/motores permanecem como alvo. Consulte VOZES_E_AMOSTRAS.md para limites e comandos atuais.

> Atualização 10/09/2026: 11 tabelas de identidade e autenticação Bearer/RBAC implementadas na revisão 0001_identity. Token persistido como SHA-256 com token_expires_at; permissions.scope_key garante unicidade global. O restante do modelo é alvo futuro. Consulte AUTENTICACAO.md para o estado atual, comandos e limitações; descrições históricas abaixo não substituem essa atualização.

08/09/2026 • Modelo proposto • 26 tabelas

Modelo proposto para implementação no Dubber, consolidado a partir do diagrama enviado e das telas de cadastro de voz e chat. Não descreve migrations já aplicadas: o código atual persiste vozes em JSON e resultados em arquivos. Nenhum banco foi criado ou alterado por esta documentação.

## Convenções

| Convenção | Definição | Regra |
| --- | --- | --- |
| Identificador numérico | id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY | Somente chave técnica local. Nunca é destino de relacionamento nem identificador recebido pela API. |
| Identificador de domínio | _id VARCHAR(255) NOT NULL UNIQUE | Gerado no servidor, preferencialmente UUID completo. Não usar os oito caracteres do cadastro legado como padrão novo. |
| Referências | *_id VARCHAR(255) | Toda referência local aponta para tabela._id, nunca tabela.id. Nulabilidade explícita em cada campo. |
| Comparação dos identificadores | Mesmo charset/collation em PK alternativa e FKs | Proposta: ASCII / ascii_bin para _id e referências, com identificadores ASCII; conteúdo textual em utf8mb4. Comparação exata e sensível a caixa. |
| Status | ENUM('active','inactive') NOT NULL DEFAULT 'active' | Em todas as 26 tabelas. Estados de processamento, entrega e armazenamento têm colunas próprias. |
| Configuração e histórico | config LONGTEXT NULL; changes_history LONGTEXT NULL | Preserva o estilo do diagrama. Conteúdo JSON validado no servidor; não guardar relações, senhas, tokens ou histórico ilimitado nesses campos. |
| Datas | created_at TIMESTAMP; updated_at TIMESTAMP NULL | UTC na conexão e na persistência; created_at com DEFAULT CURRENT_TIMESTAMP. Datas operacionais também em UTC. |
| Persistência alvo | MySQL / InnoDB | Modelo relacional proposto; verificar compatibilidade da versão instalada ao escrever migrations. Este HTML não executa SQL. |
| Escritório | office_id VARCHAR(255) NOT NULL nas entidades privadas | languages e gender são catálogos globais. permissions admite office_id NULL apenas para definições globais. |
| Retenção | Inativação lógica e ON DELETE RESTRICT | Não apagar em cascata vozes, arquivos, resultados e histórico. Exclusão física exige política explícita e ausência de referências que precisem ser preservadas. |

## Correções do diagrama

| Campo/tema | Origem | Atualizado |
| --- | --- | --- |
| office_id, user_id, profile_id, position_id, permission_id | INT / tipos misturados | VARCHAR(255) → _id do destino |
| voices.language_id / gender_id | LONGTEXT | VARCHAR(255) → languages._id / gender._id |
| user_level_id | VARCHAR(100), destino indefinido | VARCHAR(255) → user_position._id; vínculo do próprio usuário |
| voices.backing_vocal | Caminho direto de arquivo | voice_samples + media_files; ver hipótese explicada na tabela voices |
| status em vozes, idioma e gênero | Estados copiados de outros domínios | ENUM('active','inactive') |
| user_access_codes.expirtes_at | Nome com erro de digitação | expires_at |
| Campos 2fa | Mistura de número e texto | 2fa_required e 2fa_active BOOLEAN; nomes preservados e escapados em SQL |
| Escopo de profiles e relações auxiliares | Sem office_id explícito | office_id acrescentado para isolamento e FKs compostas |
| Unicidade de email/username/slugs privados | Global ou indefinida | Por escritório; catálogos globais mantêm slug global único |

## offices

Escritório/empresa que delimita os dados e a autorização.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| name | VARCHAR(255) | NÃO | — |
| slug | VARCHAR(255) | NÃO | — |
| language | VARCHAR(20) | NÃO | Código de idioma, por exemplo pt-BR; não é FK |
| currency | VARCHAR(3) | NÃO | Código de moeda, por exemplo BRL |
| address_street | VARCHAR(255) | SIM | — |
| address_number | VARCHAR(100) | SIM | — |
| address_complement | VARCHAR(100) | SIM | — |
| address_neighborhood | VARCHAR(100) | SIM | — |
| address_city | VARCHAR(100) | SIM | — |
| address_state | VARCHAR(100) | SIM | — |
| address_country | VARCHAR(100) | SIM | — |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(slug); INDEX(status)

Aplicam-se as convenções gerais.

## profiles

Perfil pessoal vinculado ao escritório.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| first_name | VARCHAR(255) | NÃO | — |
| last_name | VARCHAR(255) | NÃO | — |
| email | VARCHAR(255) | NÃO | — |
| phone | VARCHAR(40) | SIM | — |
| document_type | VARCHAR(30) | SIM | — |
| document_value | VARCHAR(100) | SIM | — |
| address_street | VARCHAR(255) | SIM | — |
| address_number | VARCHAR(100) | SIM | — |
| address_complement | VARCHAR(100) | SIM | — |
| address_neighborhood | VARCHAR(100) | SIM | — |
| address_city | VARCHAR(100) | SIM | — |
| address_state | VARCHAR(100) | SIM | — |
| address_country | VARCHAR(100) | SIM | — |
| custom_attributes | LONGTEXT | SIM | JSON validado; atributos adicionais |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, email); INDEX(office_id, status); INDEX(office_id, document_value)

office_id foi acrescentado para isolar perfis. Email normalizado antes da unicidade; permite o mesmo email em escritórios diferentes. Documentos e contatos não entram em logs.

## users

Identidade de acesso, preservando o nome users do diagrama.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_type | VARCHAR(50) | NÃO | Tipo funcional; não concede permissão por si só |
| username | VARCHAR(255) | NÃO | — |
| password | VARCHAR(255) | NÃO | Hash de senha; nunca texto puro |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, username); INDEX(office_id, status)

Cada registro pertence a um escritório. O mesmo indivíduo em outro escritório possui outra identidade/vínculo neste modelo.

## positions

Cargo do escritório.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| name | VARCHAR(255) | NÃO | — |
| slug | VARCHAR(255) | NÃO | — |
| description | TEXT | SIM | — |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, slug); INDEX(office_id, status)

Cargos pertencem a um escritório. Não existe bypass por nome de cargo. Slug customer pode existir uma vez por escritório.

## permissions

Permissão entity.action; pode ser global ou específica de escritório.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | SIM | FK offices._id; INDEX. NULL representa definição global |
| name | VARCHAR(255) | NÃO | — |
| slug | VARCHAR(255) | NÃO | — |
| description | TEXT | SIM | — |
| entity | VARCHAR(255) | NÃO | — |
| action | VARCHAR(255) | NÃO | — |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, entity, action); UNIQUE(office_id, slug); INDEX(entity, action); INDEX(office_id, status)

No MySQL, UNIQUE com office_id NULL não impede duplicatas globais. Para o catálogo global, o provisionamento deve garantir unicidade; antes de implementar, escolher índice funcional/coluna de escopo normalizada se for exigida garantia concorrente no banco. Duplicar uma definição não deve ampliar o conjunto de permissões.

## user_position

Associação de usuário e cargo.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_id | VARCHAR(255) | NÃO | FK users._id; INDEX. |
| position_id | VARCHAR(255) | NÃO | FK positions._id; INDEX. |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, user_id, position_id); INDEX(office_id, user_id, status)

office_id foi acrescentado. Usuário e cargo precisam pertencer ao mesmo escritório. Mais de um cargo por usuário é permitido.

## position_permission

Associação de cargo e permissão.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| position_id | VARCHAR(255) | NÃO | FK positions._id; INDEX. |
| permission_id | VARCHAR(255) | NÃO | FK permissions._id; INDEX. |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, position_id, permission_id); INDEX(office_id, position_id, status)

Cargo do mesmo escritório; permissão global ou do mesmo escritório. O escopo global exige validação no serviço, pois não cabe na FK composta de escritório.

## office_employees

Vínculo de funcionário com usuário, perfil e escritório.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_id | VARCHAR(255) | NÃO | FK users._id; INDEX. |
| user_level_id | VARCHAR(255) | SIM | FK user_position._id; INDEX. Cargo principal por meio do vínculo user_position |
| profile_id | VARCHAR(255) | NÃO | FK profiles._id; INDEX. |
| 2fa_required | BOOLEAN | NÃO | DEFAULT false |
| 2fa_active | BOOLEAN | NÃO | DEFAULT false |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, user_id); INDEX(office_id, status)

user_level_id aponta para user_position._id e deve ser um vínculo do próprio user_id. Campos 2fa exigem escape em SQL por começarem com número. Não autenticam sozinhos.

## user_customers

Vínculo autenticável; token de entrada da API.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_id | VARCHAR(255) | NÃO | FK users._id; INDEX. |
| user_level_id | VARCHAR(255) | SIM | FK user_position._id; INDEX. Vínculo do cargo principal |
| token | VARCHAR(255) | SIM | UNIQUE; token opaco aleatório; não expor em consultas comuns |
| profile_id | VARCHAR(255) | NÃO | FK profiles._id; INDEX. |
| 2fa_required | BOOLEAN | NÃO | DEFAULT false |
| 2fa_active | BOOLEAN | NÃO | DEFAULT false |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, user_id); UNIQUE(token); INDEX(office_id, status)

Preserva token para compatibilidade com o padrão discutido. NULL significa sem token. Comparação exata, inclusive maiúsculas/minúsculas. Token não expira automaticamente neste desenho. Perfil, usuário e user_level_id devem ser coerentes com o escritório e o usuário.

## user_access_codes

Códigos temporários de verificação.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_id | VARCHAR(255) | NÃO | FK users._id; INDEX. |
| code | VARCHAR(255) | NÃO | Hash/HMAC do código; segredo de baixa entropia requer proteção adicional e limite de tentativas |
| send_type | VARCHAR(20) | NÃO | email ou sms |
| send_to | VARCHAR(255) | NÃO | Destino; dado sensível |
| sent_at | TIMESTAMP | SIM | — |
| expires_at | TIMESTAMP | NÃO | — |
| used_at | TIMESTAMP | SIM | — |
| attempts | INT UNSIGNED | NÃO | DEFAULT 0 |
| delivery_state | ENUM('created','sent','failed') | NÃO | DEFAULT 'created' |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. INDEX(office_id, user_id, send_type); INDEX(expires_at)

Estado de utilização derivado de used_at/expires_at, separado de status. Código não é UNIQUE global: códigos curtos podem se repetir entre usuários. Verificar validade e consumir atomicamente uma vez. Não reutilizar o antigo campo expirtes_at: corrigido para expires_at.

## api_credentials

Credenciais que o backend usa para acessar provedores externos.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| name | VARCHAR(255) | NÃO | — |
| token | TEXT | NÃO | Valor criptografado recuperável somente no backend |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, name); INDEX(office_id, status)

Não autentica chamadas de entrada. Segredo não aparece em config, changes_history ou respostas. Chave de criptografia fica fora do banco.

## languages

Catálogo global de idiomas.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| name | VARCHAR(255) | NÃO | — |
| slug | VARCHAR(255) | NÃO | UNIQUE; código como pt-BR, en-US |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(slug); INDEX(status)

Substitui os estados de convite/membro presentes por engano no desenho. Idioma cadastrado não garante suporte pelo motor; o serviço valida o suporte.

## gender

Catálogo global de classificação de gênero da voz; nome singular preservado.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| name | VARCHAR(255) | NÃO | — |
| slug | VARCHAR(255) | NÃO | UNIQUE; por exemplo neutra, feminino, masculino |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(slug); INDEX(status)

Não utiliza estados sent/edited/deleted/failed. Mantém somente active/inactive.

## voices

Catálogo de vozes selecionáveis no chat.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| name | VARCHAR(255) | NÃO | — |
| language_id | VARCHAR(255) | NÃO | FK languages._id; INDEX. |
| gender_id | VARCHAR(255) | SIM | FK gender._id; INDEX. |
| description | TEXT | SIM | — |
| current_sample_id | VARCHAR(255) | SIM | FK voice_samples._id; INDEX. Amostra atual pronta e pertencente a esta voz |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. INDEX(office_id, status, language_id, gender_id); INDEX(office_id, name)

language_id e gender_id deixam de ser LONGTEXT: são VARCHAR(255) com FK para _id. backing_vocal é substituído pelos arquivos de voice_samples, assumindo que representava o áudio-base; se significava trilha de fundo, deve ser modelado separadamente. Vozes privadas por escritório; compartilhamento global não é concedido implicitamente.

## media_files

Metadados de áudio/vídeo enviado, normalizado ou gerado.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| created_by_user_id | VARCHAR(255) | SIM | FK users._id; INDEX. NULL para arquivo produzido automaticamente |
| original_name | VARCHAR(255) | NÃO | — |
| storage_key | VARCHAR(500) | NÃO | Chave interna no armazenamento privado; não uma URL pública permanente |
| mime_type | VARCHAR(100) | NÃO | — |
| size_bytes | BIGINT UNSIGNED | NÃO | — |
| duration_ms | BIGINT UNSIGNED | SIM | — |
| sample_rate | INT UNSIGNED | SIM | — |
| channels | SMALLINT UNSIGNED | SIM | — |
| checksum | VARCHAR(64) | SIM | SHA-256; não autoriza compartilhamento entre escritórios |
| storage_state | ENUM('pending','ready','failed','deleted') | NÃO | DEFAULT 'pending' |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, storage_key); INDEX(office_id, status, storage_state); INDEX(office_id, checksum)

Bytes ficam fora do banco. Validar conteúdo real, tamanho e duração no servidor. Prefixo de armazenamento inclui escritório e _id do arquivo. status=inactive impede uso; storage_state descreve disponibilidade física.

## voice_samples

Versões de amostras usadas para representar uma voz.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| voice_id | VARCHAR(255) | NÃO | FK voices._id; INDEX. |
| original_file_id | VARCHAR(255) | NÃO | FK media_files._id; INDEX. |
| normalized_file_id | VARCHAR(255) | SIM | FK media_files._id; INDEX. |
| validation_state | ENUM('pending','processing','ready','rejected') | NÃO | DEFAULT 'pending' |
| validation_result | LONGTEXT | SIM | JSON com métricas, verificações e mensagens |
| version | INT UNSIGNED | NÃO | Inteiro crescente por voz |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, voice_id, version); INDEX(office_id, voice_id, validation_state)

Amostras utilizadas em jobs não são sobrescritas. Criar outra versão para trocar áudio. validation_state=ready e status=active são exigidos para novo uso. Resultados de validação são calculados pelo servidor.

## tags

Classificações de voz, como jovem, narrador ou publicitária.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| name | VARCHAR(255) | NÃO | — |
| slug | VARCHAR(255) | NÃO | — |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, slug); INDEX(office_id, status)

Aplicam-se as convenções gerais.

## voice_tags

Relação de vozes e tags.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| voice_id | VARCHAR(255) | NÃO | FK voices._id; INDEX. |
| tag_id | VARCHAR(255) | NÃO | FK tags._id; INDEX. |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, voice_id, tag_id); INDEX(office_id, tag_id, status)

Voz e tag obrigatoriamente do mesmo escritório.

## dubbing_chats

Conversa e seleção atual de voz.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_customer_id | VARCHAR(255) | NÃO | FK user_customers._id; INDEX. |
| title | VARCHAR(255) | SIM | — |
| selected_voice_id | VARCHAR(255) | SIM | FK voices._id; INDEX. |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. INDEX(office_id, user_customer_id, status, updated_at)

user_customer_id é o proprietário. Por padrão, somente ele acessa a conversa; acesso administrativo exige permissão explícita e escopo do escritório. A voz selecionada é preferência; o job preserva a amostra efetivamente utilizada.

## dubbing_messages

Mensagem escrita, anexo ou resposta de processamento.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| chat_id | VARCHAR(255) | NÃO | FK dubbing_chats._id; INDEX. |
| user_customer_id | VARCHAR(255) | SIM | FK user_customers._id; INDEX. NULL para assistant/system; não é identidade enviada livremente pelo cliente |
| role | ENUM('user','assistant','system') | NÃO | — |
| message_type | ENUM('text','audio','video','result','error') | NÃO | — |
| content | LONGTEXT | SIM | — |
| reply_to_message_id | VARCHAR(255) | SIM | FK dubbing_messages._id; INDEX. Mensagem da mesma conversa |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. INDEX(office_id, chat_id, created_at, _id); INDEX(reply_to_message_id)

Autor e role definidos pelo servidor. Mensagem de texto exige conteúdo; áudio/vídeo exige anexo. Mensagem aceita para processamento é imutável ou gera nova mensagem ao editar.

## dubbing_message_files

Anexos de entrada e arquivos exibidos nas respostas.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| message_id | VARCHAR(255) | NÃO | FK dubbing_messages._id; INDEX. |
| media_file_id | VARCHAR(255) | NÃO | FK media_files._id; INDEX. |
| purpose | ENUM('source','result') | NÃO | — |
| sort_order | INT UNSIGNED | NÃO | DEFAULT 0 |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, message_id, media_file_id, purpose); INDEX(office_id, message_id, sort_order)

A estrutura suporta vários anexos; na primeira versão cada job processa somente um arquivo de entrada. WAV e MP3 podem coexistir numa resposta.

## dubbing_jobs

Solicitação durável de geração, conversão ou transcrição, executada por worker.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| user_customer_id | VARCHAR(255) | NÃO | FK user_customers._id; INDEX. |
| input_message_id | VARCHAR(255) | NÃO | FK dubbing_messages._id; INDEX. |
| output_message_id | VARCHAR(255) | SIM | FK dubbing_messages._id; INDEX. |
| voice_sample_id | VARCHAR(255) | SIM | FK voice_samples._id; INDEX. Obrigatória em geração/conversão; NULL permitido em transcribe |
| operation | ENUM('text_to_speech','speech_to_speech','transcribe') | NÃO | — |
| input_text | LONGTEXT | SIM | Snapshot do texto efetivamente utilizado |
| input_file_id | VARCHAR(255) | SIM | FK media_files._id; INDEX. |
| target_language_id | VARCHAR(255) | SIM | FK languages._id; INDEX. Obrigatório em geração; opcional para detecção na transcrição |
| speed | DECIMAL(5,2) | NÃO | DEFAULT 1.00; intervalo inicial 0.50 a 1.50 |
| pitch_semitones | DECIMAL(5,2) | NÃO | DEFAULT 0.00; intervalo inicial -6.00 a 6.00 |
| preserve_timing | BOOLEAN | NÃO | DEFAULT false; aplicável a speech_to_speech |
| engine | VARCHAR(100) | SIM | — |
| model_version | VARCHAR(100) | SIM | — |
| parameters | LONGTEXT | SIM | JSON imutável com configurações e formatos solicitados |
| idempotency_key | VARCHAR(100) | NÃO | Chave fornecida pelo consumidor; identidade não vem deste campo |
| request_hash | VARCHAR(64) | NÃO | SHA-256 dos dados normalizados da solicitação |
| processing_state | ENUM('queued','processing','completed','failed','cancelled') | NÃO | DEFAULT 'queued' |
| attempts | INT UNSIGNED | NÃO | DEFAULT 0 |
| locked_until | TIMESTAMP | SIM | Prazo de posse do worker; permite recuperação |
| worker_token | VARCHAR(64) | SIM | Token de posse interno; impede confirmação por worker antigo |
| started_at | TIMESTAMP | SIM | — |
| finished_at | TIMESTAMP | SIM | — |
| error_code | VARCHAR(100) | SIM | — |
| error_message | TEXT | SIM | Erro sanitizado, sem stack trace nem segredo |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, user_customer_id, idempotency_key); INDEX(status, processing_state, locked_until, created_at); INDEX(office_id, input_message_id); INDEX(office_id, user_customer_id, created_at)

Texto exige input_text e amostra pronta. Conversão exige input_file_id e amostra pronta. Transcrição exige arquivo e não exige voz. Mesmo idempotency_key com request_hash diferente retorna conflito. Worker confirma resultado somente com sua posse vigente. Cancelamento não é status=inactive. Preservar tempos é objetivo do processamento, não garantia matemática nem tradução automática.

## dubbing_job_outputs

Resultados persistidos da execução.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| job_id | VARCHAR(255) | NÃO | FK dubbing_jobs._id; INDEX. |
| media_file_id | VARCHAR(255) | NÃO | FK media_files._id; INDEX. |
| format | VARCHAR(20) | NÃO | wav, mp3, txt, json etc. |
| purpose | ENUM('audio','transcript','metadata') | NÃO | — |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, job_id, media_file_id); INDEX(office_id, job_id, purpose)

Arquivos intermediários não são publicados como resultado final. Criar saída e resposta do chat de forma idempotente.

## transcriptions

Texto reconhecido e suas versões editadas.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| job_id | VARCHAR(255) | NÃO | FK dubbing_jobs._id; INDEX. |
| source_file_id | VARCHAR(255) | NÃO | FK media_files._id; INDEX. |
| language_id | VARCHAR(255) | SIM | FK languages._id; INDEX. NULL quando não identificado/mapeado |
| text | LONGTEXT | NÃO | — |
| version | INT UNSIGNED | NÃO | DEFAULT 1; crescente por execução |
| origin | ENUM('recognized','edited') | NÃO | — |
| edited_by_user_id | VARCHAR(255) | SIM | FK users._id; INDEX. |
| previous_transcription_id | VARCHAR(255) | SIM | FK transcriptions._id; INDEX. Versão anterior da mesma transcrição/execução |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, job_id, version); INDEX(office_id, source_file_id)

Edição cria nova versão e não altera a entrada de jobs já aceitos. Um novo job recebe o texto escolhido como snapshot.

## dubbing_segments

Trechos temporais de transcrição e síntese sincronizada.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| job_id | VARCHAR(255) | NÃO | FK dubbing_jobs._id; INDEX. |
| transcription_id | VARCHAR(255) | SIM | FK transcriptions._id; INDEX. |
| sequence | INT UNSIGNED | NÃO | — |
| source_start_ms | BIGINT UNSIGNED | NÃO | — |
| source_end_ms | BIGINT UNSIGNED | NÃO | — |
| recognized_text | LONGTEXT | SIM | — |
| synthesis_text | LONGTEXT | SIM | — |
| output_file_id | VARCHAR(255) | SIM | FK media_files._id; INDEX. |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, job_id, sequence); INDEX(office_id, transcription_id, sequence)

source_end_ms >= source_start_ms. Milissegundos inteiros. Se preenchida, transcription_id deve pertencer ao mesmo job. Guarda trechos realmente usados na execução; editar a transcrição não reescreve segmentos executados.

## project_audio_links

Vínculo do áudio ao projeto que pertence à aplicação SiPlug.

| Coluna | Tipo | Nulo | Restrição / significado |
| --- | --- | --- | --- |
| id | INT UNSIGNED | NÃO | PK; AUTO_INCREMENT; nunca é destino de FK |
| _id | VARCHAR(255) | NÃO | UNIQUE; identificador opaco gerado no servidor |
| office_id | VARCHAR(255) | NÃO | FK offices._id; INDEX. |
| external_project_id | VARCHAR(255) | NÃO | Referência externa a projects._id da SiPlug; sem FK entre bancos |
| media_file_id | VARCHAR(255) | NÃO | FK media_files._id; INDEX. |
| created_by_user_id | VARCHAR(255) | NÃO | FK users._id; INDEX. |
| config | LONGTEXT | SIM | JSON validado pela aplicação; sem segredos |
| changes_history | LONGTEXT | SIM | JSON de alterações sanitizado; não é fila nem histórico ilimitado |
| status | ENUM('active','inactive') | NÃO | DEFAULT 'active'; disponibilidade lógica |
| created_at | TIMESTAMP | NÃO | DEFAULT CURRENT_TIMESTAMP; UTC |
| updated_at | TIMESTAMP | SIM | Atualizado pela aplicação; UTC |

**Índices:** PRIMARY KEY(id); UNIQUE(_id); índice em cada FK. UNIQUE(office_id, external_project_id, media_file_id); INDEX(office_id, external_project_id, status)

A SiPlug valida existência e autorização do projeto. O Dubber valida escritório e arquivo local. Não duplica cadastro de projetos. Sincronização precisa tratar projeto removido/inacessível.

## integridade e índices

Em todas as tabelas: PRIMARY KEY(id), UNIQUE(_id). Cada FK local recebe índice. Não criar FK para id numérico.

Nas tabelas privadas, acrescentar UNIQUE(office_id, _id). Para relações entre duas tabelas privadas, preferir FOREIGN KEY (office_id, campo_id) REFERENCES destino(office_id, _id), reforçando no banco que os registros pertencem ao mesmo escritório. Essa regra complementa as referências simples listadas no dicionário.

office_id referencia offices._id. Relações com languages e gender usam FK simples para _id. Relações com permissions globais usam FK simples e validação de escopo no serviço. external_project_id é referência externa, não FK física.

user_level_id deve indicar user_position do mesmo usuário. current_sample_id deve indicar voice_samples da mesma voz. reply_to_message_id e output_message_id precisam apontar para a mesma conversa da solicitação. Essas regras exigem validações específicas, além de pertencer ao escritório.

As FKs devem usar ON UPDATE RESTRICT e ON DELETE RESTRICT. Identificadores são imutáveis. Referências opcionais aceitam NULL, não string vazia; preencher exige recurso válido.

Criar voices com current_sample_id NULL, criar voice_samples e depois adicionar a FK de current_sample_id. Ao cadastrar, inserir voz, amostra e atualizar current_sample_id quando a amostra estiver pronta. Relações circulares opcionais são adicionadas após a criação das tabelas.

Índices descritos são parte da proposta. Evitar duplicar um índice simples já coberto pelo prefixo de outro índice; confirmar consultas e limites de tamanho na implementação.

Inativar pai impede novas operações dependentes mesmo quando filhos ainda estão ativos. FKs verificam existência, não status: essa regra pertence aos serviços.

## autenticação, autorização e segredos

Preserva users, user_customers, office_employees, user_position, positions, position_permission e permissions. Não introduz service_clients nem substitui a autenticação pela tabela api_credentials.

O backend da SiPlug envia Authorization: Bearer com credencial provisionada no Dubber. A API resolve token → user_customers → users → offices e valida todos os status antes do caso de uso. office_id e identidade autorizada são calculados no servidor.

As permissões são resolvidas por user_position → positions → position_permission → permissions, com vínculos ativos e escopo correto. Sugestão inicial: voice.create/read/update, dubbing_chat.create/read/update, dubbing.generate/read/cancel, dubbing.download e dubbing_project.link.

Usar credencial dedicada da integração por escritório ou outro provisionamento explícito no mesmo modelo. Um único token não pode escolher livremente outro escritório no JSON. O modelo não acrescenta uma delegação multiempresa implícita.

users.password guarda hash de senha; api_credentials.token guarda credencial externa criptografada. user_customers.token é mantido no formato de compatibilidade do padrão existente: token opaco direto, comparação exata, UNIQUE e rotação. Isso não equivale a hash em repouso nem expiração automática. Qualquer evolução exige alterar emissão, persistência e validação em conjunto.

user_access_codes.code representa hash/HMAC do código temporário, com estratégia de comparação definida na implementação. Limitar tentativas e proteger códigos de baixa entropia; sent_at, expires_at e used_at controlam entrega e consumo.

Campos 2fa não comprovam um fluxo de segundo fator implementado. Nunca retornar tokens, senhas, códigos, worker_token ou chaves de armazenamento internas indiscriminadamente; token novo é entregue somente no fluxo de emissão autorizado.

Vozes, arquivos, jobs e chats são privados do escritório. Chats também têm proprietário; usuários comuns acessam os próprios chats, e acesso administrativo requer permissão explícita. Download aplica a mesma autorização, inclusive quando houver URL temporária.

## fluxo do chat e processamento

1. Adicionar voz: cadastrar media_files, voices e voice_samples; validar e normalizar o áudio fora da transação longa. Quando a amostra estiver pronta, registrar normalized_file_id e current_sample_id.

2. Criar conversa: resolver actor, registrar dubbing_chats com user_customer_id e selecionar uma voz pronta do escritório.

3. Texto: criar dubbing_messages com texto. O job text_to_speech captura input_text, voice_sample_id e parâmetros. Em áudio/vídeo: registrar arquivo e anexo source; criar job speech_to_speech com input_file_id e amostra escolhida.

4. Aceitar: validar voz, formato, limites, idioma suportado e permissões. Em transação curta, criar mensagem e job com idempotency_key/request_hash. Reenvio igual retorna o job existente; conteúdo diferente com a mesma chave retorna conflito. Responder 202 com _id do job.

5. Worker: reivindicar job atomicamente, registrar worker_token/locked_until, incrementar attempts e processar fora da transação. A fila pode apenas avisar que há trabalho; o banco mantém o estado durável.

6. Conversão: registrar transcriptions e dubbing_segments quando houver reconhecimento/ajuste temporal. Definir separadamente a política de tradução; escolher outro idioma não traduz o texto por si só.

7. Concluir: salvar arquivo privado, registrar media_files e dubbing_job_outputs, criar resposta assistant/result e seus anexos, vincular output_message_id e marcar completed. Fazer a publicação no banco em uma transação idempotente, condicionada à posse vigente do worker.

8. Consultar e baixar: API devolve estado e metadados autorizados; o frontend SiPlug apresenta o resultado. Salvar no projeto cria project_audio_links após validação do projeto na SiPlug.

9. Falha/cancelamento: guardar erro sanitizado e finished_at, preservar entrada para auditoria. Retentativas precisam reutilizar o mesmo job sem duplicar a resposta final. Cancelamento é processing_state=cancelled; status=inactive é disponibilidade lógica.

10. Banco e armazenamento não compartilham transação: usar arquivos temporários, estados pending/ready e rotina de recuperação/limpeza de órfãos. Não marcar completed antes de os arquivos finais estarem disponíveis.

## organização no padrão SiPlug

Requisição → middleware actor/permission → controller → DTO de entrada → caso de uso → serviços das entidades → interfaces de repositório → implementações de repositório.

Consultas não ficam diretamente no caso de uso. Exemplos: FindVoiceByUniqueIdService, CreateDubbingMessageService, CreateDubbingJobService e FindDubbingJobByUniqueIdService usam seus repositórios.

Motores de transcrição, síntese e conversão existentes em Python são adaptadores de processamento. Não precisam ser reescritos em outra linguagem para seguir a arquitetura.

Configuração capturada em parameters e a referência voice_sample_id tornam cada execução reproduzível dentro das limitações do motor. Mudanças de voz, chat ou transcrição não alteram o snapshot de jobs antigos.

## implantação e migração futura

Este documento é especificação, não migration aplicada. Não remove frontend, não executa bootstrap, não modifica dados, não cria endpoints nem afirma que o Dubber atual já implementa esta estrutura.

Ordem sugerida: offices; profiles/users; positions/permissions; user_position/position_permission; office_employees/user_customers/user_access_codes/api_credentials; languages/gender; media_files; voices sem FK current_sample_id; voice_samples; adicionar FK current_sample_id; tags/voice_tags; dubbing_chats/dubbing_messages; dubbing_message_files; dubbing_jobs; dubbing_job_outputs/transcriptions; dubbing_segments; project_audio_links. Autorrelacionamentos opcionais podem ser adicionados após criar a tabela.

Importação legada: cada voice.json vira voices + voice_samples + media_files. Gerar _id completo e manter mapa entre ID antigo e novo. Atribuir escritório explicitamente; não inferir pelo nome de pasta. Revalidar arquivos e preservar resultados existentes antes de migrar.

O diretório local de projetos contém execuções, não comprova cadastro de projetos da SiPlug. Importar arquivos encontrados com origem conhecida, sem inventar usuário, texto, job ou vínculo externo ausentes nos metadados.

Antes das migrations finais: confirmar o significado de backing_vocal, a versão do MySQL e o mapeamento de projetos externos. A proposta já define vozes privadas; catálogo compartilhado exige modelo explícito de concessão.

Validação de implementação: todas as FKs para _id; rejeição de referências entre escritórios; autorização de proprietário; estados active/inactive; idempotência concorrente; recuperação do worker; criação de resultado única; download autorizado; edição de voz sem alterar histórico; tratamento de entrada por texto e áudio.

## Mapa completo de referências

| Origem | Destino | Tipo | Aceita NULL |
| --- | --- | --- | --- |
| profiles.office_id | offices._id | VARCHAR(255) | NÃO |
| users.office_id | offices._id | VARCHAR(255) | NÃO |
| positions.office_id | offices._id | VARCHAR(255) | NÃO |
| permissions.office_id | offices._id | VARCHAR(255) | SIM |
| user_position.office_id | offices._id | VARCHAR(255) | NÃO |
| user_position.user_id | users._id | VARCHAR(255) | NÃO |
| user_position.position_id | positions._id | VARCHAR(255) | NÃO |
| position_permission.office_id | offices._id | VARCHAR(255) | NÃO |
| position_permission.position_id | positions._id | VARCHAR(255) | NÃO |
| position_permission.permission_id | permissions._id | VARCHAR(255) | NÃO |
| office_employees.office_id | offices._id | VARCHAR(255) | NÃO |
| office_employees.user_id | users._id | VARCHAR(255) | NÃO |
| office_employees.user_level_id | user_position._id | VARCHAR(255) | SIM |
| office_employees.profile_id | profiles._id | VARCHAR(255) | NÃO |
| user_customers.office_id | offices._id | VARCHAR(255) | NÃO |
| user_customers.user_id | users._id | VARCHAR(255) | NÃO |
| user_customers.user_level_id | user_position._id | VARCHAR(255) | SIM |
| user_customers.profile_id | profiles._id | VARCHAR(255) | NÃO |
| user_access_codes.office_id | offices._id | VARCHAR(255) | NÃO |
| user_access_codes.user_id | users._id | VARCHAR(255) | NÃO |
| api_credentials.office_id | offices._id | VARCHAR(255) | NÃO |
| voices.office_id | offices._id | VARCHAR(255) | NÃO |
| voices.language_id | languages._id | VARCHAR(255) | NÃO |
| voices.gender_id | gender._id | VARCHAR(255) | SIM |
| voices.current_sample_id | voice_samples._id | VARCHAR(255) | SIM |
| media_files.office_id | offices._id | VARCHAR(255) | NÃO |
| media_files.created_by_user_id | users._id | VARCHAR(255) | SIM |
| voice_samples.office_id | offices._id | VARCHAR(255) | NÃO |
| voice_samples.voice_id | voices._id | VARCHAR(255) | NÃO |
| voice_samples.original_file_id | media_files._id | VARCHAR(255) | NÃO |
| voice_samples.normalized_file_id | media_files._id | VARCHAR(255) | SIM |
| tags.office_id | offices._id | VARCHAR(255) | NÃO |
| voice_tags.office_id | offices._id | VARCHAR(255) | NÃO |
| voice_tags.voice_id | voices._id | VARCHAR(255) | NÃO |
| voice_tags.tag_id | tags._id | VARCHAR(255) | NÃO |
| dubbing_chats.office_id | offices._id | VARCHAR(255) | NÃO |
| dubbing_chats.user_customer_id | user_customers._id | VARCHAR(255) | NÃO |
| dubbing_chats.selected_voice_id | voices._id | VARCHAR(255) | SIM |
| dubbing_messages.office_id | offices._id | VARCHAR(255) | NÃO |
| dubbing_messages.chat_id | dubbing_chats._id | VARCHAR(255) | NÃO |
| dubbing_messages.user_customer_id | user_customers._id | VARCHAR(255) | SIM |
| dubbing_messages.reply_to_message_id | dubbing_messages._id | VARCHAR(255) | SIM |
| dubbing_message_files.office_id | offices._id | VARCHAR(255) | NÃO |
| dubbing_message_files.message_id | dubbing_messages._id | VARCHAR(255) | NÃO |
| dubbing_message_files.media_file_id | media_files._id | VARCHAR(255) | NÃO |
| dubbing_jobs.office_id | offices._id | VARCHAR(255) | NÃO |
| dubbing_jobs.user_customer_id | user_customers._id | VARCHAR(255) | NÃO |
| dubbing_jobs.input_message_id | dubbing_messages._id | VARCHAR(255) | NÃO |
| dubbing_jobs.output_message_id | dubbing_messages._id | VARCHAR(255) | SIM |
| dubbing_jobs.voice_sample_id | voice_samples._id | VARCHAR(255) | SIM |
| dubbing_jobs.input_file_id | media_files._id | VARCHAR(255) | SIM |
| dubbing_jobs.target_language_id | languages._id | VARCHAR(255) | SIM |
| dubbing_job_outputs.office_id | offices._id | VARCHAR(255) | NÃO |
| dubbing_job_outputs.job_id | dubbing_jobs._id | VARCHAR(255) | NÃO |
| dubbing_job_outputs.media_file_id | media_files._id | VARCHAR(255) | NÃO |
| transcriptions.office_id | offices._id | VARCHAR(255) | NÃO |
| transcriptions.job_id | dubbing_jobs._id | VARCHAR(255) | NÃO |
| transcriptions.source_file_id | media_files._id | VARCHAR(255) | NÃO |
| transcriptions.language_id | languages._id | VARCHAR(255) | SIM |
| transcriptions.edited_by_user_id | users._id | VARCHAR(255) | SIM |
| transcriptions.previous_transcription_id | transcriptions._id | VARCHAR(255) | SIM |
| dubbing_segments.office_id | offices._id | VARCHAR(255) | NÃO |
| dubbing_segments.job_id | dubbing_jobs._id | VARCHAR(255) | NÃO |
| dubbing_segments.transcription_id | transcriptions._id | VARCHAR(255) | SIM |
| dubbing_segments.output_file_id | media_files._id | VARCHAR(255) | SIM |
| project_audio_links.office_id | offices._id | VARCHAR(255) | NÃO |
| project_audio_links.media_file_id | media_files._id | VARCHAR(255) | NÃO |
| project_audio_links.created_by_user_id | users._id | VARCHAR(255) | NÃO |

Referência externa: project_audio_links.external_project_id → projects._id da SiPlug; VARCHAR(255); sem FK física entre bancos.
