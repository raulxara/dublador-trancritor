"""Owned chats, immutable messages and durable job submissions."""

from uuid import NAMESPACE_URL, uuid5

import sqlalchemy as sa
from alembic import op

revision = "0003_chat_jobs"
down_revision = "0002_voices"
branch_labels = None
depends_on = None
STATEMENTS = [
    """CREATE TABLE dubbing_chats (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_customer_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `title` VARCHAR(255) NULL,
  `selected_voice_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_dubbing_chats_office_id FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_user_customer_id (`user_customer_id`),
  CONSTRAINT fk_dubbing_chats_user_customer_id FOREIGN KEY (office_id,`user_customer_id`)
 REFERENCES user_customers (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_selected_voice_id (`selected_voice_id`),
  CONSTRAINT fk_dubbing_chats_selected_voice_id FOREIGN KEY (office_id,`selected_voice_id`)
 REFERENCES voices (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_query_0 (office_id, user_customer_id, status, updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE dubbing_messages (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `chat_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_customer_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `role` ENUM('user','assistant','system') NOT NULL,
  `message_type` ENUM('text','audio','video','result','error') NOT NULL,
  `content` LONGTEXT NULL,
  `reply_to_message_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  UNIQUE KEY uq_message_chat (office_id,chat_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_dubbing_messages_office_id FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_chat_id (`chat_id`),
  CONSTRAINT fk_dubbing_messages_chat_id FOREIGN KEY (office_id,`chat_id`)
 REFERENCES dubbing_chats (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_user_customer_id (`user_customer_id`),
  CONSTRAINT fk_dubbing_messages_user_customer_id FOREIGN KEY (office_id,`user_customer_id`)
 REFERENCES user_customers (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_reply_to_message_id (`reply_to_message_id`),
  CONSTRAINT fk_dubbing_messages_reply_to_message_id FOREIGN KEY (office_id,chat_id,reply_to_message_id)
 REFERENCES dubbing_messages (office_id,chat_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_query_0 (office_id, chat_id, created_at, _id),
  KEY ix_query_1 (reply_to_message_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE dubbing_message_files (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `message_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `media_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `purpose` ENUM('source','result') NOT NULL,
  `sort_order` INT UNSIGNED NOT NULL DEFAULT 0,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_dubbing_message_files_office_id FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_message_id (`message_id`),
  CONSTRAINT fk_dubbing_message_files_message_id FOREIGN KEY (office_id,`message_id`)
 REFERENCES dubbing_messages (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_media_file_id (`media_file_id`),
  CONSTRAINT fk_dubbing_message_files_media_file_id FOREIGN KEY (office_id,`media_file_id`)
 REFERENCES media_files (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  UNIQUE KEY uq_1 (office_id, message_id, media_file_id, purpose),
  KEY ix_query_0 (office_id, message_id, sort_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE dubbing_jobs (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_customer_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `input_message_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `output_message_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `voice_sample_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `operation` ENUM('text_to_speech','speech_to_speech','transcribe') NOT NULL,
  `input_text` LONGTEXT NULL,
  `input_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `target_language_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `speed` DECIMAL(5,2) NOT NULL DEFAULT 1.00,
  `pitch_semitones` DECIMAL(5,2) NOT NULL DEFAULT 0.00,
  `preserve_timing` BOOLEAN NOT NULL DEFAULT false,
  `engine` VARCHAR(100) NULL,
  `model_version` VARCHAR(100) NULL,
  `parameters` LONGTEXT NULL,
  `idempotency_key` VARCHAR(100) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `request_hash` VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `processing_state` ENUM('queued','processing','completed','failed','cancelled') NOT NULL DEFAULT 'queued',
  `attempts` INT UNSIGNED NOT NULL DEFAULT 0,
  `locked_until` TIMESTAMP NULL,
  `worker_token` VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `started_at` TIMESTAMP NULL,
  `finished_at` TIMESTAMP NULL,
  `error_code` VARCHAR(100) NULL,
  `error_message` TEXT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY uq_scope (office_id,_id),
  KEY ix_office_id (`office_id`),
  CONSTRAINT fk_dubbing_jobs_office_id FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_user_customer_id (`user_customer_id`),
  CONSTRAINT fk_dubbing_jobs_user_customer_id FOREIGN KEY (office_id,`user_customer_id`)
 REFERENCES user_customers (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_input_message_id (`input_message_id`),
  CONSTRAINT fk_dubbing_jobs_input_message_id FOREIGN KEY (office_id,`input_message_id`)
 REFERENCES dubbing_messages (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_output_message_id (`output_message_id`),
  CONSTRAINT fk_dubbing_jobs_output_message_id FOREIGN KEY (office_id,`output_message_id`)
 REFERENCES dubbing_messages (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_voice_sample_id (`voice_sample_id`),
  CONSTRAINT fk_dubbing_jobs_voice_sample_id FOREIGN KEY (office_id,`voice_sample_id`)
 REFERENCES voice_samples (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_input_file_id (`input_file_id`),
  CONSTRAINT fk_dubbing_jobs_input_file_id FOREIGN KEY (office_id,`input_file_id`)
 REFERENCES media_files (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  KEY ix_target_language_id (`target_language_id`),
  CONSTRAINT fk_dubbing_jobs_target_language_id FOREIGN KEY (`target_language_id`)
 REFERENCES languages (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
  UNIQUE KEY uq_1 (office_id, user_customer_id, idempotency_key),
  KEY ix_query_0 (status, processing_state, locked_until, created_at),
  KEY ix_query_1 (office_id, input_message_id),
  KEY ix_query_2 (office_id, user_customer_id, created_at),
  CHECK (speed BETWEEN 0.50 AND 1.50),
  CHECK (pitch_semitones BETWEEN -6.00 AND 6.00)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
]
PERMISSIONS = [
    ("dubbing_chat", "create"),
    ("dubbing_chat", "read"),
    ("dubbing_chat", "update"),
    ("dubbing", "generate"),
    ("dubbing", "read"),
    ("dubbing", "cancel"),
    ("dubbing", "download"),
]


def upgrade():
    for statement in STATEMENTS:
        op.execute(statement)
    connection = op.get_bind()
    for entity, action in PERMISSIONS:
        slug = entity + "." + action
        identifier = str(uuid5(NAMESPACE_URL, "siplug-dubber/permission/" + slug))
        connection.execute(
            sa.text("""INSERT INTO permissions (_id,name,slug,entity,action)
            VALUES (:id,:slug,:slug,:entity,:action)"""),
            dict(id=identifier, slug=slug, entity=entity, action=action),
        )
        connection.execute(
            sa.text("""INSERT INTO position_permission (_id,office_id,position_id,permission_id)
            SELECT UUID(),pos.office_id,pos._id,:permission FROM positions pos
            WHERE pos.slug='admin' AND pos.status='active' AND EXISTS (
                SELECT 1 FROM position_permission pp JOIN permissions p ON p._id=pp.permission_id
                WHERE pp.position_id=pos._id AND pp.office_id=pos.office_id AND pp.status='active'
                AND p.status='active' AND p.entity='user' AND p.action='update'
                AND (p.office_id IS NULL OR p.office_id=pos.office_id))"""),
            dict(permission=identifier),
        )


def downgrade():
    connection = op.get_bind()
    for entity, action in PERMISSIONS:
        identifier = str(uuid5(NAMESPACE_URL, "siplug-dubber/permission/" + entity + "." + action))
        connection.execute(sa.text("DELETE FROM position_permission WHERE permission_id=:id"), dict(id=identifier))
        connection.execute(sa.text("DELETE FROM permissions WHERE _id=:id"), dict(id=identifier))
    for table in ["dubbing_jobs", "dubbing_message_files", "dubbing_messages", "dubbing_chats"]:
        op.execute("DROP TABLE " + table)
