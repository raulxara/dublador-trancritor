# Autenticação e migrations — SiPlug Dubber

## Implementado

A revisão Alembic `0001_identity` cria offices, profiles, users, positions, permissions, user_position, position_permission, office_employees, user_customers, user_access_codes e api_credentials. Todas as referências apontam para `_id` VARCHAR(255), com comparação exata e FKs compostas para escritório/proprietário. O status continua active/inactive. As outras 15 tabelas do modelo entram nas etapas de áudio e chat.

Diferenças de segurança em relação ao modelo inicial: `user_customers.token` armazena SHA-256 do token, `token_expires_at` limita a validade a 30 dias na emissão e NULL revoga o acesso; `permissions.scope_key` é coluna gerada usada para impedir duplicatas globais. O hash não é aceito como credencial: o cliente envia o token original de 64 caracteres hexadecimais. Tokens não são JWTs.

## Preparar o banco

```sh
docker compose up -d --build --wait
docker compose exec app python -m alembic upgrade head
docker compose exec app python -m alembic current
```

Não há seed automático ou usuário padrão. O health/readiness continua verificando processo/conexão, não substitui a conferência de `alembic current`. O downgrade é destrutivo e só deve ser testado em banco descartável.

## Primeiro acesso

Execute uma única vez em instalação vazia, substituindo os dados ilustrativos pelos reais:

```sh
docker compose exec app python -m app.console.bootstrap_access \
  --office-name "Minha empresa" --office-slug "minha-empresa" \
  --first-name "Ana" --last-name "Silva" \
  --email "ana@example.test" --username "ana.admin"
```

A senha é solicitada sem eco e salva com scrypt; não existe login público por senha nesta etapa. O comando retorna IDs, token e validade: guarde a saída em local seguro para configurar o backend SiPlug. Não coloque esse token de integração no frontend. Escritório, perfil, usuário, vínculo, cargo e permissões são criados em transação. O lock impede dois bootstraps simultâneos. O comando recusa banco que já tenha escritório, usuário ou vínculo; não serve para recuperar acesso perdido.

O cadastro real inicial não foi executado automaticamente. Os usuários dos testes existem somente no banco isolado.

## Requisições

O backend SiPlug envia `Authorization: Bearer <token>`. Query, JSON, officeId ou permissions enviados pelo cliente não definem o actor. A dependência de autenticação roda antes do controller, chama o serviço e o repositório; valida vínculo, usuário, escritório e perfil ativos e prazo do token. As permissões vêm de vínculos, cargos e concessões ativos do mesmo escritório, aceitando definições globais ou locais. Não existe bypass por nome de cargo.

| Endpoint | Regra |
| --- | --- |
| GET /api/v1/health e /ready | Públicos |
| GET /api/v1/auth/context | Bearer e catalog.read |
| POST /api/v1/users/{user_id}/access-token | Bearer e user.update; alvo ativo no escritório do actor |
| DELETE /api/v1/auth/access-token | Bearer; revoga o próprio vínculo |

401: token ausente, malformado, inválido, expirado, revogado ou identidade inativa. 403: permissão ausente. 404: usuário alvo ausente ou de outro escritório. 503: falha de persistência, sem mensagem do driver. Respostas de contexto e credencial usam Cache-Control no-store.

A rotação devolve o token novo uma vez na resposta e invalida o antigo nas próximas requisições. Quem tem user.update pode emitir tokens para qualquer usuário ativo do próprio escritório: é uma permissão administrativa. A revogação/expiração não cancela requisições já autorizadas. Depois de perder/revogar o último token administrativo, será necessário um procedimento de recuperação local, ainda não implementado; bootstrap não contorna essa restrição.

## Validação

```sh
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from app
docker compose -f docker-compose.test.yml down -v
```

O Compose de teste usa banco efêmero, sem portas publicadas e sem rede compartilhada. Não execute os testes de integração apontando para o banco de desenvolvimento: eles provisionam e alteram registros de teste. `DUBBER_INTEGRATION=1` é exclusivo desse ambiente. Os testes cobrem bootstrap único, hash, FKs, contexto, status, permissões, expiração, rotação, revogação e isolamento entre escritórios.

Ainda não estão implementados: login por senha/2FA, entrega de códigos, recuperação de acesso, gestão de usuários/cargos, uso de credenciais externas, vozes, chat e workers. As respectivas tabelas ou campos não significam que esses fluxos estejam disponíveis.
