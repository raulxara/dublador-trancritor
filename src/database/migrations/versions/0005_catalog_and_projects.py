"""Voice classification and external project references."""

from uuid import NAMESPACE_URL, uuid5

import sqlalchemy as sa
from alembic import op

revision = "0005_catalog_and_projects"
down_revision = "0004_processing_results"
branch_labels = None
depends_on = None
STATEMENTS = [
    """CREATE TABLE tags (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
`_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
`office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`name` VARCHAR(255) NOT NULL,
`slug` VARCHAR(255) NOT NULL,
`config` LONGTEXT NULL,
`changes_history` LONGTEXT NULL,
`status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
`created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
`updated_at` TIMESTAMP NULL,
UNIQUE KEY uq_scope (office_id,_id),
UNIQUE KEY uq_slug (office_id,slug),
KEY ix_status (office_id,status),
FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE voice_tags (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
`_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
`office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`voice_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`tag_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`config` LONGTEXT NULL,
`changes_history` LONGTEXT NULL,
`status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
`created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
`updated_at` TIMESTAMP NULL,
UNIQUE KEY uq_scope (office_id,_id),
UNIQUE KEY uq_voice_tag (office_id,voice_id,tag_id),
KEY ix_tag (office_id,tag_id,status),
FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`voice_id`)
 REFERENCES voices (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`tag_id`)
 REFERENCES tags (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE project_audio_links (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
`_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
`office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`external_project_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`media_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`created_by_user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`config` LONGTEXT NULL,
`changes_history` LONGTEXT NULL,
`status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
`created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
`updated_at` TIMESTAMP NULL,
UNIQUE KEY uq_scope (office_id,_id),
UNIQUE KEY uq_project_media (office_id,external_project_id,media_file_id),
KEY ix_project (office_id,external_project_id,status),
FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`media_file_id`)
 REFERENCES media_files (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`created_by_user_id`)
 REFERENCES users (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
]
STATEMENTS.append("CREATE INDEX ix_media_storage_key ON media_files (storage_key)")

PERMISSIONS = [
    ("tag", "read"),
    ("tag", "update"),
    ("project_audio", "read"),
    ("project_audio", "update"),
    ("transcription", "update"),
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
    op.execute("DROP INDEX ix_media_storage_key ON media_files")
    connection = op.get_bind()
    for entity, action in PERMISSIONS:
        identifier = str(uuid5(NAMESPACE_URL, "siplug-dubber/permission/" + entity + "." + action))
        connection.execute(sa.text("DELETE FROM position_permission WHERE permission_id=:id"), dict(id=identifier))
        connection.execute(sa.text("DELETE FROM permissions WHERE _id=:id"), dict(id=identifier))
    for table in ["project_audio_links", "voice_tags", "tags"]:
        op.execute("DROP TABLE " + table)
