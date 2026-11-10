-- Idempotent: safe to re-run. Applied by the `migrate` service in the compose files.

CREATE TABLE IF NOT EXISTS job (
    id               uuid        PRIMARY KEY,
    request_id       text        NOT NULL,
    surface          text        NOT NULL,
    content_id       text        NOT NULL,
    content_version  integer     NOT NULL CHECK (content_version >= 1),
    status           text        NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'dead')),
    payload          jsonb       NOT NULL,
    attempts         integer     NOT NULL DEFAULT 0,
    next_attempt_at  timestamptz NOT NULL DEFAULT now(),
    lease_until      timestamptz,
    error_code       text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT job_content_key UNIQUE (surface, content_id, content_version)
);

CREATE INDEX IF NOT EXISTS job_claim_idx ON job (status, next_attempt_at);

CREATE TABLE IF NOT EXISTS decision (
    job_id          uuid        PRIMARY KEY REFERENCES job (id) ON DELETE CASCADE,
    label           text        NOT NULL CHECK (label IN ('allow', 'hold', 'block')),
    category        text,
    reason          text        NOT NULL,
    items           jsonb       NOT NULL DEFAULT '[]'::jsonb,
    policy_version  text        NOT NULL,
    model_versions  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS outbox (
    id               bigserial   PRIMARY KEY,
    job_id           uuid        NOT NULL REFERENCES job (id) ON DELETE CASCADE,
    callback_url     text        NOT NULL CHECK (char_length(callback_url) <= 2048),
    body             jsonb       NOT NULL,
    status           text        NOT NULL CHECK (status IN ('queued', 'sent', 'dead')),
    attempts         integer     NOT NULL DEFAULT 0,
    next_attempt_at  timestamptz NOT NULL DEFAULT now(),
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS outbox_claim_idx ON outbox (status, next_attempt_at);

-- SigLIP2 image embeddings, written with the decision. real[] for now; D5 moves it to pgvector.
CREATE TABLE IF NOT EXISTS image_embedding (
    job_id      uuid        NOT NULL REFERENCES job (id) ON DELETE CASCADE,
    media_id    text        NOT NULL,
    model       text,
    embedding   real[]      NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, media_id)
);
