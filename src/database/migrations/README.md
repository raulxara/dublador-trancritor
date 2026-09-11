# Migrations

Alembic: `python -m alembic upgrade head`, dentro do container. Revisão `0001_identity` cria 11 tabelas de identidade, autorização e credenciais; as 15 tabelas de áudio/chat serão adicionadas nas próximas etapas.

Referências VARCHAR(255), ASCII/ascii_bin, apontam para `_id`; FKs compostas preservam escritório e proprietário do cargo principal. Status active/inactive. DDL é explícito e congelado na revisão; não depende de modelos mutáveis nem cria tabelas no boot.

`user_customers.token` contém SHA-256 do token aleatório, com `token_expires_at` obrigatório para aceitar acesso. NULL revoga o acesso. `permissions.scope_key` gerada garante unicidade também para permissões globais.

MySQL faz commit implícito de DDL. Execute uma migration por vez; em falha parcial, inspecione o schema antes de repetir. Downgrade remove tabelas e dados e foi testado exclusivamente em banco isolado; não o use para desfazer dados reais.

A revisão 0002_voices acrescenta cinco tabelas de vozes/amostras, catálogos iniciais e concessões explícitas de permissão para administradores existentes. Veja docs/VOZES_E_AMOSTRAS.md na raiz do projeto.

0003_chat_jobs cria quatro tabelas de chat/mensagens/jobs e sete permissões explícitas para administradores existentes. Proprietário e escritório são verificados pelos serviços. O worker ainda não é executado.
