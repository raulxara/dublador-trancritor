# Persistência

`authorization/` contém os repositórios SQLAlchemy de autenticação e bootstrap. SQL parametrizado fica nesta camada; controllers e use cases não acessam o banco. O bootstrap usa transação e lock MySQL para impedir provisionamentos concorrentes.

As tabelas são criadas por migrations Alembic explícitas. Mapeamentos de vozes e chat serão implementados com seus repositórios nas próximas etapas.
