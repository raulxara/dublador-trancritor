# Cadastro de clientes no Dubber

`POST /api/v1/register-user` exige `Authorization: Bearer <token>` e permissão `user.register` na empresa do solicitante. Não é uma rota pública de autoinscrição. O backend Siplug chama esta rota com sua credencial de provisionamento.

```json
{
  "firstName": "Cliente",
  "lastName": "Exemplo",
  "email": "cliente@example.com",
  "phone": "5511999999999",
  "username": "cliente@example.com",
  "password": "uma-senha-forte-com-12-ou-mais-caracteres",
  "documentType": "cpf",
  "documentValue": "CPF normalizado"
}
```

`officeId` é opcional e, se informado, deve corresponder ao `_id` da empresa autenticada. CPF é armazenado; a validação dos dígitos pertence ao cadastro público no backend. Senha aceita de 12 a 1024 caracteres; telefone em formato internacional, com + opcional. Campos extras como permissions, userType, status ou positionSlug são rejeitados.

Uma transação cria profiles, users (user_type=customer), user_position e user_customers. A posição customer da mesma empresa é reutilizada ou criada sem permissões. Uma posição inativa não é reativada automaticamente. Todas as referências são pelo _id. A rota vincula à empresa existente, não cria outra empresa e não envia códigos de verificação; a confirmação dos contatos continua no backend Siplug.

Retorno HTTP 201: success e data com officeId, profileId, userId, userCustomerId, userPositionId, positionId, assignedPositionSlug, userType, status, token e expiresAt. Não guardar o retorno nos logs. Cache-Control: no-store.

A senha usa scrypt no padrão existente; o token aleatório tem validade de 30 dias e somente SHA-256 é salvo no Dubber. O backend deve criptografar o token original ao persistir sua cópia. A renovação existente é POST /api/v1/users/{userId}/access-token, protegida por user.update.

Email ou username repetidos na mesma empresa retornam 409, sem duplicação nem rotação automática do token. Chamadas concorrentes são serializadas por empresa. Isto não é replay idempotente da resposta: em caso de resposta perdida, o integrador deve conciliar o cadastro, não criar novas identidades. O mesmo email em outra empresa não conflita.

Instalações novas incluem user.register no bootstrap administrativo. Instalações existentes precisam cadastrar a permissão entity=user, action=register e vinculá-la à posição da credencial administrativa, na mesma empresa. Nenhuma permissão de contas existentes é alterada automaticamente. Os clientes recebem somente as permissões configuradas na posição customer.

Testes: docker compose -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from app. Usar exclusivamente o banco efêmero desse Compose; nunca ativar DUBBER_INTEGRATION no banco de desenvolvimento.
