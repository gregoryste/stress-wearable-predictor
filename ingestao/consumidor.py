import json
import logging
from datetime import datetime

import paho.mqtt.client as mqtt
import psycopg
from psycopg.types.json import Jsonb

TOPICO = "wearable/+/sinais"
INSERT = """
    INSERT INTO demo.mensagens_mqtt
        (topico, usuario, timestamp_mensagem, payload)
    VALUES (%s, %s, %s, %s)
    RETURNING id
"""


def ler_mensagem(topico, dados):
    mensagem = json.loads(dados)
    if not isinstance(mensagem, dict):
        raise ValueError("esperado um objeto JSON")
    usuario = mensagem.get("usuario")
    if not isinstance(usuario, str) or not usuario or topico != f"wearable/{usuario}/sinais":
        raise ValueError("usuario ausente ou diferente do tópico")
    if not isinstance(mensagem.get("sinais"), dict) or not mensagem["sinais"]:
        raise ValueError("sinais ausentes ou vazios")
    timestamp = mensagem.get("timestamp")
    if not isinstance(timestamp, str):
        raise ValueError("timestamp ausente")
    instante = datetime.fromisoformat(timestamp)
    if instante.tzinfo is None:
        raise ValueError("timestamp precisa incluir o fuso horário")
    return usuario, instante, mensagem


def ao_conectar(cliente, dados, flags, motivo, propriedades):
    if motivo.is_failure:
        logging.error("Conexão recusada: %s", motivo)
        return
    cliente.subscribe(TOPICO, qos=1)


def ao_assinar(cliente, dados, mid, motivos, propriedades):
    if not motivos or any(motivo.is_failure for motivo in motivos):
        raise RuntimeError("O broker recusou a assinatura")
    logging.info("Escutando %s", TOPICO)


def ao_receber(cliente, banco, evento):
    try:
        usuario, instante, mensagem = ler_mensagem(evento.topic, evento.payload)
    except (ValueError, UnicodeError) as erro:
        logging.warning("Mensagem inválida em %s: %s", evento.topic, erro)
    else:
        with banco.transaction():
            registro = banco.execute(
                INSERT, (evento.topic, usuario, instante, Jsonb(mensagem))
            ).fetchone()[0]
        logging.info("Gravada id=%s usuario=%s topico=%s", registro, usuario, evento.topic)
    if evento.qos:
        cliente.ack(evento.mid, evento.qos)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with psycopg.connect(autocommit=True, connect_timeout=5) as banco:
        cliente = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id="ingestao-demo",
            clean_session=False, userdata=banco, manual_ack=True,
        )
        cliente.on_connect = ao_conectar
        cliente.on_subscribe = ao_assinar
        cliente.on_message = ao_receber
        cliente.connect_async("mqtt", 1883)
        cliente.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
