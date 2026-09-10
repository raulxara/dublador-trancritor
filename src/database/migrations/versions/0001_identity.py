"""Identity and authorization; references use public string identifiers."""

from alembic import op

revision = "0001_identity"
down_revision = None
branch_labels = None
depends_on = None

STATEMENTS = [
    """CREATE TABLE `offices` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `name` VARCHAR(255) NOT NULL,
  `slug` VARCHAR(255) NOT NULL,
  `language` VARCHAR(20) NOT NULL,
  `currency` VARCHAR(3) NOT NULL,
  `address_street` VARCHAR(255) NULL,
  `address_number` VARCHAR(100) NULL,
  `address_complement` VARCHAR(100) NULL,
  `address_neighborhood` VARCHAR(100) NULL,
  `address_city` VARCHAR(100) NULL,
  `address_state` VARCHAR(100) NULL,
  `address_country` VARCHAR(100) NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_1` (`slug`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `profiles` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `first_name` VARCHAR(255) NOT NULL,
  `last_name` VARCHAR(255) NOT NULL,
  `email` VARCHAR(255) NOT NULL,
  `phone` VARCHAR(40) NULL,
  `document_type` VARCHAR(30) NULL,
  `document_value` VARCHAR(100) NULL,
  `address_street` VARCHAR(255) NULL,
  `address_number` VARCHAR(100) NULL,
  `address_complement` VARCHAR(100) NULL,
  `address_neighborhood` VARCHAR(100) NULL,
  `address_city` VARCHAR(100) NULL,
  `address_state` VARCHAR(100) NULL,
  `address_country` VARCHAR(100) NULL,
  `custom_attributes` LONGTEXT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_profiles_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `users` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_type` VARCHAR(50) NOT NULL,
  `username` VARCHAR(255) NOT NULL,
  `password` VARCHAR(255) NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_users_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `positions` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `slug` VARCHAR(255) NOT NULL,
  `description` TEXT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_positions_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`slug`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `permissions` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `name` VARCHAR(255) NOT NULL,
  `slug` VARCHAR(255) NOT NULL,
  `description` TEXT NULL,
  `entity` VARCHAR(255) NOT NULL,
  `action` VARCHAR(255) NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_permissions_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`entity`,`action`),
  UNIQUE KEY `uq_2` (`office_id`,`slug`),
  `scope_key` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin GENERATED ALWAYS AS (COALESCE(office_id,'')) STORED,
  UNIQUE KEY `uq_global_action` (`scope_key`,`entity`,`action`),
  UNIQUE KEY `uq_global_slug` (`scope_key`,`slug`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `user_position` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `position_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  UNIQUE KEY `uq_owner` (`office_id`,`user_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_user_position_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  KEY `ix_user_id` (`user_id`),
  CONSTRAINT `fk_user_position_user_id`
    FOREIGN KEY (`office_id`,`user_id`)
    REFERENCES `users` (`office_id`,`_id`) ON DELETE RESTRICT,
  KEY `ix_position_id` (`position_id`),
  CONSTRAINT `fk_user_position_position_id`
    FOREIGN KEY (`office_id`,`position_id`)
    REFERENCES `positions` (`office_id`,`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`user_id`,`position_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `position_permission` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `position_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `permission_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_position_permission_office_id`
    FOREIGN KEY (`office_id`)
    REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  KEY `ix_position_id` (`position_id`),
  CONSTRAINT `fk_position_permission_position_id`
    FOREIGN KEY (`office_id`,`position_id`)
    REFERENCES `positions` (`office_id`,`_id`) ON DELETE RESTRICT,
  KEY `ix_permission_id` (`permission_id`),
  CONSTRAINT `fk_position_permission_permission_id`
    FOREIGN KEY (`permission_id`)
    REFERENCES `permissions` (`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`position_id`,`permission_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `office_employees` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_level_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `profile_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `2fa_required` BOOLEAN NOT NULL DEFAULT false,
  `2fa_active` BOOLEAN NOT NULL DEFAULT false,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_office_employees_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  KEY `ix_user_id` (`user_id`),
  CONSTRAINT `fk_office_employees_user_id`
    FOREIGN KEY (`office_id`,`user_id`)
    REFERENCES `users` (`office_id`,`_id`) ON DELETE RESTRICT,
  KEY `ix_user_level_id` (`user_level_id`),
  CONSTRAINT `fk_office_employees_user_level_id`
    FOREIGN KEY (`office_id`,`user_id`,`user_level_id`)
    REFERENCES `user_position` (`office_id`,`user_id`,`_id`) ON DELETE RESTRICT,
  KEY `ix_profile_id` (`profile_id`),
  CONSTRAINT `fk_office_employees_profile_id`
    FOREIGN KEY (`office_id`,`profile_id`)
    REFERENCES `profiles` (`office_id`,`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `user_customers` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_level_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `token` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NULL,
  `profile_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `2fa_required` BOOLEAN NOT NULL DEFAULT false,
  `2fa_active` BOOLEAN NOT NULL DEFAULT false,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  `token_expires_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_user_customers_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  KEY `ix_user_id` (`user_id`),
  CONSTRAINT `fk_user_customers_user_id`
    FOREIGN KEY (`office_id`,`user_id`)
    REFERENCES `users` (`office_id`,`_id`) ON DELETE RESTRICT,
  KEY `ix_user_level_id` (`user_level_id`),
  CONSTRAINT `fk_user_customers_user_level_id`
    FOREIGN KEY (`office_id`,`user_id`,`user_level_id`)
    REFERENCES `user_position` (`office_id`,`user_id`,`_id`) ON DELETE RESTRICT,
  KEY `ix_profile_id` (`profile_id`),
  CONSTRAINT `fk_user_customers_profile_id`
    FOREIGN KEY (`office_id`,`profile_id`)
    REFERENCES `profiles` (`office_id`,`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`user_id`),
  UNIQUE KEY `uq_2` (`token`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `user_access_codes` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `user_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `code` VARCHAR(255) NOT NULL,
  `send_type` VARCHAR(20) NOT NULL,
  `send_to` VARCHAR(255) NOT NULL,
  `sent_at` TIMESTAMP NULL,
  `expires_at` TIMESTAMP NOT NULL,
  `used_at` TIMESTAMP NULL,
  `attempts` INT UNSIGNED NOT NULL DEFAULT 0,
  `delivery_state` ENUM('created','sent','failed') NOT NULL DEFAULT 'created',
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_user_access_codes_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  KEY `ix_user_id` (`user_id`),
  CONSTRAINT `fk_user_access_codes_user_id`
    FOREIGN KEY (`office_id`,`user_id`)
    REFERENCES `users` (`office_id`,`_id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
    """CREATE TABLE `api_credentials` (
  `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  `_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  `office_id` VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `token` TEXT NOT NULL,
  `config` LONGTEXT NULL,
  `changes_history` LONGTEXT NULL,
  `status` ENUM('active','inactive') NOT NULL DEFAULT 'active',
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NULL,
  UNIQUE KEY `uq_scope` (`office_id`,`_id`),
  KEY `ix_office_id` (`office_id`),
  CONSTRAINT `fk_api_credentials_office_id` FOREIGN KEY (`office_id`) REFERENCES `offices` (`_id`) ON DELETE RESTRICT,
  UNIQUE KEY `uq_1` (`office_id`,`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""",
]


INDEXES = [
    "CREATE INDEX ix_offices_0 ON `offices` (`status`)",
    "CREATE INDEX ix_profiles_0 ON `profiles` (`office_id`,`status`)",
    "CREATE INDEX ix_profiles_1 ON `profiles` (`office_id`,`document_value`)",
    "CREATE INDEX ix_users_0 ON `users` (`office_id`,`status`)",
    "CREATE INDEX ix_positions_0 ON `positions` (`office_id`,`status`)",
    "CREATE INDEX ix_permissions_0 ON `permissions` (`entity`,`action`)",
    "CREATE INDEX ix_permissions_1 ON `permissions` (`office_id`,`status`)",
    "CREATE INDEX ix_user_position_0 ON `user_position` (`office_id`,`user_id`,`status`)",
    "CREATE INDEX ix_position_permission_0 ON `position_permission` (`office_id`,`position_id`,`status`)",
    "CREATE INDEX ix_office_employees_0 ON `office_employees` (`office_id`,`status`)",
    "CREATE INDEX ix_user_customers_0 ON `user_customers` (`office_id`,`status`)",
    "CREATE INDEX ix_user_access_codes_0 ON `user_access_codes` (`office_id`,`user_id`,`send_type`)",
    "CREATE INDEX ix_user_access_codes_1 ON `user_access_codes` (`expires_at`)",
    "CREATE INDEX ix_api_credentials_0 ON `api_credentials` (`office_id`,`status`)",
]


def upgrade():
    for statement in STATEMENTS + INDEXES:
        op.execute(statement)


def downgrade():
    for table in [
        "api_credentials",
        "user_access_codes",
        "user_customers",
        "office_employees",
        "position_permission",
        "user_position",
        "permissions",
        "positions",
        "users",
        "profiles",
        "offices",
    ]:
        op.execute(f"DROP TABLE `{table}`")
