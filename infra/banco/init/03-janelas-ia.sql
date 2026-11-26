CREATE SCHEMA IF NOT EXISTS demo;

CREATE TABLE IF NOT EXISTS demo.progresso_janelas (
    usuario text PRIMARY KEY,
    ultimo_id_consumido bigint NOT NULL
);

CREATE TABLE IF NOT EXISTS demo.predicoes (
    usuario text NOT NULL,
    primeiro_id bigint NOT NULL,
    ultimo_id bigint NOT NULL,
    fim_janela timestamptz NOT NULL,
    features jsonb NOT NULL,
    resposta jsonb NOT NULL,
    gravado_em timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (usuario, ultimo_id)
);

CREATE INDEX IF NOT EXISTS mensagens_mqtt_usuario_id
    ON demo.mensagens_mqtt (usuario, id);
