"""Private execution results, transcripts and timed segments."""

from alembic import op

revision = "0004_processing_results"
down_revision = "0003_chat_jobs"
branch_labels = None
depends_on = None
STATEMENTS = [
    """CREATE TABLE dubbing_job_outputs (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
`_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
`office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`job_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`media_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`format` VARCHAR(20) NOT NULL,
`purpose` ENUM('audio','transcript','metadata') NOT NULL,
`config` LONGTEXT NULL,
`changes_history` LONGTEXT NULL,
`status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
`created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
`updated_at` TIMESTAMP NULL,
UNIQUE KEY uq_scope (office_id,_id),
UNIQUE KEY uq_output (office_id,job_id,media_file_id),
KEY ix_purpose (office_id,job_id,purpose),
FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`job_id`)
 REFERENCES dubbing_jobs (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`media_file_id`)
 REFERENCES media_files (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE transcriptions (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
`_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
`office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`job_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`source_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`language_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
`text` LONGTEXT NOT NULL,
`version` INT UNSIGNED NOT NULL DEFAULT 1,
`origin` ENUM('recognized','edited') NOT NULL,
`edited_by_user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
`previous_transcription_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
`config` LONGTEXT NULL,
`changes_history` LONGTEXT NULL,
`status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
`created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
`updated_at` TIMESTAMP NULL,
UNIQUE KEY uq_scope (office_id,_id),
UNIQUE KEY uq_version (office_id,job_id,version),
UNIQUE KEY uq_job_transcription (office_id,job_id,_id),
FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`job_id`)
 REFERENCES dubbing_jobs (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`source_file_id`)
 REFERENCES media_files (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (`language_id`)
 REFERENCES languages (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`edited_by_user_id`)
 REFERENCES users (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,job_id,`previous_transcription_id`)
 REFERENCES transcriptions (office_id,job_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE dubbing_segments (
id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
`_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
`office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`job_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
`transcription_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
`sequence` INT UNSIGNED NOT NULL,
`source_start_ms` BIGINT UNSIGNED NOT NULL,
`source_end_ms` BIGINT UNSIGNED NOT NULL,
`recognized_text` LONGTEXT NULL,
`synthesis_text` LONGTEXT NULL,
`output_file_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
`config` LONGTEXT NULL,
`changes_history` LONGTEXT NULL,
`status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
`created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
`updated_at` TIMESTAMP NULL,
UNIQUE KEY uq_scope (office_id,_id),
UNIQUE KEY uq_sequence (office_id,job_id,`sequence`),
CHECK (source_end_ms>=source_start_ms),
FOREIGN KEY (`office_id`)
 REFERENCES offices (_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`job_id`)
 REFERENCES dubbing_jobs (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,job_id,`transcription_id`)
 REFERENCES transcriptions (office_id,job_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
FOREIGN KEY (office_id,`output_file_id`)
 REFERENCES media_files (office_id,_id) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
]


def upgrade():
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    for table in ("dubbing_segments", "transcriptions", "dubbing_job_outputs"):
        op.execute("DROP TABLE " + table)
