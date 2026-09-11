"""Voice catalog and immutable sample versions."""

from uuid import NAMESPACE_URL, uuid5

import sqlalchemy as sa
from alembic import op

revision = "0002_voices"
down_revision = "0001_identity"
branch_labels = None
depends_on = None
STATEMENTS = [
    """CREATE TABLE languages (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `name` VARCHAR(255) NOT NULL,
  `slug` VARCHAR(255) NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_1 (slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE gender (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `name` VARCHAR(255) NOT NULL,
  `slug` VARCHAR(255) NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_1 (slug)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE voices (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `language_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `gender_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `description` TEXT NULL,
  `current_sample_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_voices_office_id FOREIGN KEY (`office_id`)
    REFERENCES offices (`_id`) ON DELETE RESTRICT,
  KEY ix_language_id (`language_id`),
  CONSTRAINT fk_voices_language_id FOREIGN KEY (`language_id`)
    REFERENCES languages (`_id`) ON DELETE RESTRICT,
  KEY ix_gender_id (`gender_id`),
  CONSTRAINT fk_voices_gender_id FOREIGN KEY (`gender_id`)
    REFERENCES gender (`_id`) ON DELETE RESTRICT,
  KEY ix_current_sample_id (`current_sample_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE media_files (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `created_by_user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `original_name` VARCHAR(255) NOT NULL,
  `storage_key` VARCHAR(500) NOT NULL,
  `mime_type` VARCHAR(100) NOT NULL,
  `size_bytes` BIGINT UNSIGNED NOT NULL,
  `duration_ms` BIGINT UNSIGNED NULL,
  `sample_rate` INT UNSIGNED NULL,
  `channels` SMALLINT UNSIGNED NULL,
  `checksum` VARCHAR(64) NULL,
  `storage_state` ENUM('pending','ready','failed','deleted') NOT NULL DEFAULT 'pending',
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_media_files_office_id FOREIGN KEY (`office_id`)
    REFERENCES offices (`_id`) ON DELETE RESTRICT,
  KEY ix_created_by_user_id (`created_by_user_id`),
  CONSTRAINT fk_media_files_created_by_user_id FOREIGN KEY (office_id,`created_by_user_id`)
    REFERENCES users (office_id,_id) ON DELETE RESTRICT,
  UNIQUE KEY uq_1 (office_id, storage_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE voice_samples (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `voice_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `original_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `normalized_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `validation_state` ENUM('pending','processing','ready','rejected') NOT NULL DEFAULT 'pending',
  `validation_result` LONGTEXT NULL,
  `version` INT UNSIGNED NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  UNIQUE KEY uq_voice_sample (office_id,voice_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_voice_samples_office_id FOREIGN KEY (`office_id`)
    REFERENCES offices (`_id`) ON DELETE RESTRICT,
  KEY ix_voice_id (`voice_id`),
  CONSTRAINT fk_voice_samples_voice_id FOREIGN KEY (office_id,`voice_id`)
    REFERENCES voices (office_id,_id) ON DELETE RESTRICT,
  KEY ix_original_file_id (`original_file_id`),
  CONSTRAINT fk_voice_samples_original_file_id FOREIGN KEY (office_id,`original_file_id`)
    REFERENCES media_files (office_id,_id) ON DELETE RESTRICT,
  KEY ix_normalized_file_id (`normalized_file_id`),
  CONSTRAINT fk_voice_samples_normalized_file_id FOREIGN KEY (office_id,`normalized_file_id`)
    REFERENCES media_files (office_id,_id) ON DELETE RESTRICT,
  UNIQUE KEY uq_1 (office_id, voice_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """ALTER TABLE voices ADD CONSTRAINT fk_voices_current_sample FOREIGN KEY (office_id,_id,current_sample_id)
    REFERENCES voice_samples (office_id,voice_id,_id) ON DELETE RESTRICT""",
    """CREATE INDEX ix_languages_0 ON languages (status)""",
    """CREATE INDEX ix_gender_0 ON gender (status)""",
    """CREATE INDEX ix_voices_0 ON voices (office_id, status, language_id, gender_id)""",
    """CREATE INDEX ix_voices_1 ON voices (office_id, name)""",
    """CREATE INDEX ix_media_files_0 ON media_files (office_id, status, storage_state)""",
    """CREATE INDEX ix_media_files_1 ON media_files (office_id, checksum)""",
    """CREATE INDEX ix_voice_samples_0 ON voice_samples (office_id, voice_id, validation_state)""",
]


def upgrade():
    for statement in STATEMENTS:
        op.execute(statement)
    connection = op.get_bind()
    for table, rows in [
        ("languages", [("pt-BR", "Português (Brasil)"), ("en", "English"), ("es", "Español")]),
        ("gender", [("neutra", "Neutra"), ("feminino", "Feminino"), ("masculino", "Masculino")]),
    ]:
        for slug, name in rows:
            connection.execute(
                sa.text(f"INSERT INTO {table} (_id,name,slug) VALUES (:id,:name,:slug)"),
                dict(id=str(uuid5(NAMESPACE_URL, "siplug-dubber/" + table + "/" + slug)), name=name, slug=slug),
            )
    for action in ["read", "register", "update"]:
        permission = str(uuid5(NAMESPACE_URL, "siplug-dubber/permission/voice." + action))
        connection.execute(
            sa.text("INSERT INTO permissions (_id,name,slug,entity,action) VALUES (:id,:slug,:slug,'voice',:action)"),
            dict(id=permission, slug="voice." + action, action=action),
        )
        # Explicit migration of bootstrap administrators; no runtime administrator bypass.
        connection.execute(
            sa.text("""INSERT INTO position_permission (_id,office_id,position_id,permission_id)
            SELECT UUID(),pos.office_id,pos._id,:permission FROM positions pos
            WHERE pos.slug='admin' AND pos.status='active' AND EXISTS (
                SELECT 1 FROM position_permission pp JOIN permissions p ON p._id=pp.permission_id
                WHERE pp.position_id=pos._id AND pp.office_id=pos.office_id AND pp.status='active'
                AND p.status='active' AND p.entity='user' AND p.action='update'
                AND (p.office_id IS NULL OR p.office_id=pos.office_id))"""),
            dict(permission=permission),
        )


def downgrade():
    connection = op.get_bind()
    for action in ["read", "register", "update"]:
        permission = str(uuid5(NAMESPACE_URL, "siplug-dubber/permission/voice." + action))
        connection.execute(sa.text("DELETE FROM position_permission WHERE permission_id=:id"), dict(id=permission))
        connection.execute(sa.text("DELETE FROM permissions WHERE _id=:id"), dict(id=permission))
    op.execute("ALTER TABLE voices DROP FOREIGN KEY fk_voices_current_sample")
    for table in ["voice_samples", "media_files", "voices", "gender", "languages"]:
        op.execute("DROP TABLE " + table)
