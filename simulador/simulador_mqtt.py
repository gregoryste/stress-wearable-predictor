import json
import os
import time
from datetime import datetime, timezone
import paho.mqtt.client as mqtt


def gerar_mensagem_exemplo(usuario: str = "S2") -> dict:
    """Versão falsa (v1): devolve sinais fixos no formato do contrato."""
    return {
        "usuario": usuario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sinais": {
            "bvp": [0.12, 0.15, 0.11],
            "eda": [1.42, 1.44],
            "temp": [33.5, 33.5],
            "acc": [[0.01, -0.02, 0.98]],
        },
        "contexto": {
            "horario": "14:00",
            "ruido_db": 45.0,
        },
    }


def main():
    broker = os.getenv("MQTT_BROKER", "localhost")
    porta = int(os.getenv("MQTT_PORT", "1883"))
    usuario = os.getenv("WEARABLE_USER", "S2")
    topico = f"wearable/{usuario}/sinais"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker, porta, 60)

    print(f"Publicando em {broker}:{porta} no tópico '{topico}' (Ctrl+C para parar)")
    while True:
        mensagem = gerar_mensagem_exemplo(usuario)
        client.publish(topico, json.dumps(mensagem))
        print("Enviado:", mensagem["timestamp"])
        time.sleep(2)


if __name__ == "__main__":
    main()