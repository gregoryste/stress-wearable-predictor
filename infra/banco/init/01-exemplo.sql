CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE SCHEMA IF NOT EXISTS demo;
CREATE TABLE demo.sinais (
    instante timestamptz NOT NULL,
    participante text NOT NULL,
    temperatura_c numeric NOT NULL,
    PRIMARY KEY (participante, instante)
);
SELECT create_hypertable('demo.sinais', 'instante');
INSERT INTO demo.sinais VALUES
    ('2026-11-23 12:00:00+00', 'exemplo', 32.1),
    ('2026-11-23 12:00:01+00', 'exemplo', 32.2);
