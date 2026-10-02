"""Registration behavior against the isolated MySQL schema."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text


def check_registration(client, headers, container, office_id):
    path = "/api/v1/register-user"
    payload = dict(
        firstName="Cliente",
        lastName="Teste",
        email="customer@example.test",
        phone="5511999999999",
        username="customer@example.test",
        password="NewCustomerPassword!123",
        documentType="cpf",
        documentValue="12345678909",
    )
    assert client.post(path, json=payload).status_code == 401
    assert client.post(path, headers=headers, json=payload | {"officeId": "foreign-office"}).status_code == 403
    for extra in [{"userType": "admin"}, {"permissions": ["user.register"]}, {"positionSlug": "admin"}]:
        assert client.post(path, headers=headers, json=payload | extra).status_code == 422
    for invalid in [{"email": "invalid"}, {"phone": "bad"}, {"password": "short"}]:
        result = client.post(path, headers=headers, json=payload | invalid)
        assert result.status_code == 422
        assert payload["password"] not in result.text
    response = client.post(path, headers=headers, json=payload)
    assert response.status_code == 201, response.text
    assert response.headers["cache-control"] == "no-store"
    data = response.json()["data"]
    assert data["officeId"] == office_id
    assert data["assignedPositionSlug"] == "customer"
    assert payload["password"] not in response.text
    customer_headers = {"Authorization": "Bearer " + data["token"]}
    # New role has no implicit administrative or business permissions.
    assert client.post(path, headers=customer_headers, json=payload).status_code == 403
    with container.engine.connect() as connection:
        row = (
            connection.execute(text("SELECT * FROM user_customers WHERE _id=:id"), {"id": data["userCustomerId"]})
            .mappings()
            .one()
        )
        assert row["token"] == hashlib.sha256(data["token"].encode()).hexdigest()
        assert row["user_id"] == data["userId"]
        assert row["profile_id"] == data["profileId"]
        assert row["user_level_id"] == data["userPositionId"]
        password = connection.execute(
            text("SELECT password FROM users WHERE _id=:id"), {"id": data["userId"]}
        ).scalar_one()
        _, n, block, parallel, salt, digest = password.split("$")
        assert (
            hashlib.scrypt(
                payload["password"].encode(), salt=salt.encode(), n=int(n), r=int(block), p=int(parallel)
            ).hex()
            == digest
        )
    assert client.post(path, headers=headers, json=payload).status_code == 409
    # Concurrent retries cannot create more than one account.
    concurrent = payload | {"username": "parallel", "email": "parallel@example.test"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(lambda _: client.post(path, headers=headers, json=concurrent).status_code, range(2)))
    assert sorted(statuses) == [201, 409]
    # An error on the final insert must roll back earlier profile/user/role associations.
    with container.engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TRIGGER reject_test_registration BEFORE INSERT ON user_customers "
                "FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='test rollback'"
            )
        )
    try:
        failed = payload | {"username": "rollback", "email": "rollback@example.test"}
        assert client.post(path, headers=headers, json=failed).status_code == 503
        with container.engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM users WHERE username='rollback'")).scalar_one() == 0
            assert (
                connection.execute(
                    text("SELECT COUNT(*) FROM profiles WHERE email='rollback@example.test'")
                ).scalar_one()
                == 0
            )
    finally:
        with container.engine.begin() as connection:
            connection.execute(text("DROP TRIGGER reject_test_registration"))
    # Existing customer role reused; grant read permission explicitly and check issued token works.
    with container.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO position_permission (_id,office_id,position_id,permission_id) "
                "SELECT 'registration-test-grant',:office,:position,_id FROM permissions "
                "WHERE office_id=:office AND entity='catalog' AND action='read'"
            ),
            {"office": office_id, "position": data["positionId"]},
        )
    assert client.get("/api/v1/auth/context", headers=customer_headers).status_code == 200
