"""Gera predições a partir dos lotes de um segundo gravados pelo consumidor MQTT."""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import psycopg
from psycopg.types.json import Jsonb

from contratos.api import Predicao
from shared.features import DURACAO_JANELA_S, FREQUENCIAS_HZ, extrair_features, juntar_mensagens


def lote_de_um_segundo(mensagem):
    sinais = mensagem.get("sinais") if isinstance(mensagem, dict) else None
    return isinstance(sinais, dict) and all(
        isinstance(sinais.get(nome), list) and len(sinais[nome]) == hz
        for nome, hz in FREQUENCIAS_HZ.items()
    )


def prever(usuario, fim, features):
    corpo = {
        "usuario": usuario, "fim_janela": fim.isoformat(),
        "duracao_s": DURACAO_JANELA_S, "features": features,
    }
    pedido = Request(os.environ["API_PREVER_URL"], data=json.dumps(corpo).encode(),
                     headers={"Content-Type": "application/json"})
    with urlopen(pedido, timeout=15) as resposta:
        predicao = Predicao.model_validate_json(resposta.read())
    resposta_fim = predicao.fim_janela
    if isinstance(resposta_fim, str):
        resposta_fim = datetime.fromisoformat(resposta_fim)
    if predicao.usuario != usuario or resposta_fim != fim:
        raise ValueError("A resposta da API não corresponde à janela enviada")
    return predicao.model_dump(mode="json")


def processar_usuario(banco, usuario):
    # O contador e a predição são confirmados na mesma transação.
    with banco.transaction():
        banco.execute("""
            INSERT INTO demo.progresso_janelas VALUES (%s, 0)
            ON CONFLICT DO NOTHING
        """, (usuario,))
        cursor = banco.execute("""
            SELECT ultimo_id_consumido FROM demo.progresso_janelas
            WHERE usuario = %s FOR UPDATE
        """, (usuario,)).fetchone()[0]

        linhas = banco.execute("""
            SELECT id, timestamp_mensagem, payload FROM demo.mensagens_mqtt
            WHERE usuario = %s AND id > %s ORDER BY id LIMIT %s
        """, (usuario, cursor, DURACAO_JANELA_S)).fetchall()
        if not linhas:
            return False

        # Um lote de teste ou um relógio que volta rompe a sequência.
        corte = 0
        for i, (_, instante, mensagem) in enumerate(linhas):
            if i and instante <= linhas[i - 1][1]:
                corte = i
            if not lote_de_um_segundo(mensagem):
                corte = i + 1
        if corte:
            avancar_ate = linhas[corte - 1][0]
        elif len(linhas) < DURACAO_JANELA_S:
            return False
        else:
            primeiro, ultimo = linhas[0][0], linhas[-1][0]
            fim = linhas[-1][1]
            try:
                sinais = juntar_mensagens([linha[2] for linha in linhas])
                features = extrair_features(sinais).model_dump()
            except ValueError as erro:
                logging.warning("Janela descartada %s [%s..%s]: %s", usuario, primeiro, ultimo, erro)
            else:
                resposta = prever(usuario, fim, features)
                banco.execute("""
                    INSERT INTO demo.predicoes
                        (usuario, primeiro_id, ultimo_id, fim_janela, features, resposta)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (usuario, ultimo_id) DO NOTHING
                """, (usuario, primeiro, ultimo, fim, Jsonb(features), Jsonb(resposta)))
                logging.info("Predição %s [%s..%s]: %s", usuario, primeiro, ultimo, resposta["estado"])
            avancar_ate = ultimo

        banco.execute("""
            UPDATE demo.progresso_janelas SET ultimo_id_consumido = %s WHERE usuario = %s
        """, (avancar_ate, usuario))
        return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with psycopg.connect(autocommit=True) as banco:
        banco.execute(Path("/app/schema.sql").read_text())
        logging.info("Aguardando lotes de um segundo em demo.mensagens_mqtt")
        while True:
            usuarios = banco.execute("SELECT DISTINCT usuario FROM demo.mensagens_mqtt").fetchall()
            avancou = False
            houve_erro = False
            for (usuario,) in usuarios:
                try:
                    if processar_usuario(banco, usuario):
                        avancou = True
                except (OSError, ValueError) as erro:
                    logging.warning("Falha ao prever para %s; janela pendente: %s", usuario, erro)
                    houve_erro = True
            if houve_erro or not avancou:
                time.sleep(3)


if __name__ == "__main__":
    main()
