.PHONY: up down test lint logs
up:
	docker compose up -d --build --wait
down:
	docker compose down
test:
	docker compose run --rm --no-deps app python -m pytest -q -p no:cacheprovider
lint:
	docker compose run --rm --no-deps app ruff check --no-cache app tests
	docker compose run --rm --no-deps app ruff format --check app tests
logs:
	docker compose logs --tail=100 -f app db
