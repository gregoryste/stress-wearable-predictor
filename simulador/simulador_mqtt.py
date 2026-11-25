import json
import os
import pickle
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


def gerar_mensagem_exemplo(usuario: str = "S2") -> dict:
    """Fallback v1: devolve sinais fixos no formato do contrato."""
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


def carregar_sinais_wrist(caminho_pkl: str) -> dict:
    """Carrega os sinais da pulseira Empatica E4 de um ficheiro .pkl do WESAD."""
    with open(caminho_pkl, "rb") as ficheiro:
        dados = pickle.load(ficheiro, encoding="latin1")
    return dados["signal"]["wrist"]


def calcular_total_segundos(wrist: dict) -> int:
    """Calcula quantos segundos completos sincronizados existem nos 4 sensores."""
    return min(
        len(wrist["BVP"]) // 64,
        len(wrist["ACC"]) // 32,
        len(wrist["EDA"]) // 4,
        len(wrist["TEMP"]) // 4,
    )


def montar_payload_wesad(
    usuario: str,
    wrist: dict,
    segundo_idx: int,
    horario: str = "14:00",
    ruido_db: float = 45.0,
) -> dict:
    """Fatia 1 segundo sincronizado (BVP 64Hz, ACC 32Hz, EDA 4Hz, TEMP 4Hz)."""
    return {
        "usuario": usuario,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sinais": {
            "bvp": wrist["BVP"][segundo_idx * 64 : (segundo_idx + 1) * 64]
            .flatten()
            .tolist(),
            "eda": wrist["EDA"][segundo_idx * 4 : (segundo_idx + 1) * 4]
            .flatten()
            .tolist(),
            "temp": wrist["TEMP"][segundo_idx * 4 : (segundo_idx + 1) * 4]
            .flatten()
            .tolist(),
            "acc": wrist["ACC"][segundo_idx * 32 : (segundo_idx + 1) * 32].tolist(),
        },
        "contexto": {
            "horario": horario,
            "ruido_db": ruido_db,
        },
    }


def main():
    broker = os.getenv("MQTT_BROKER", "localhost")
    porta = int(os.getenv("MQTT_PORT", "1883"))
    usuario = os.getenv("WEARABLE_USER", "S2")
    caminho_pkl = os.getenv("WESAD_PKL_PATH", "")
    velocidade = float(os.getenv("SIM_SPEED", "1.0"))
    horario = os.getenv("SIM_HORARIO", "14:00")
    ruido_db = float(os.getenv("SIM_RUIDO_DB", "45.0"))
    topico = f"wearable/{usuario}/sinais"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker, porta, 60)
    client.loop_start()

    try:
        if caminho_pkl and os.path.exists(caminho_pkl):
            print(f"Carregando WESAD de '{caminho_pkl}'...")
            wrist = carregar_sinais_wrist(caminho_pkl)
            total_segundos = calcular_total_segundos(wrist)
            print(
                f"Reproduzindo {total_segundos}s em {broker}:{porta} "
                f"(tópico '{topico}', velocidade {velocidade}x) — Ctrl+C para parar"
            )

            for seg in range(total_segundos):
                mensagem = montar_payload_wesad(
                    usuario=usuario,
                    wrist=wrist,
                    segundo_idx=seg,
                    horario=horario,
                    ruido_db=ruido_db,
                )
                info = client.publish(topico, json.dumps(mensagem), qos=1)
                info.wait_for_publish(timeout=5)
                print(
                    f"Enviado segundo {seg + 1}/{total_segundos}: {mensagem['timestamp']}"
                )
                time.sleep(1.0 / velocidade)
        else:
            print(
                f"WESAD_PKL_PATH não definido ou não encontrado. "
                f"Publicando exemplo em {broker}:{porta} ('{topico}') — Ctrl+C para parar"
            )
            while True:
                mensagem = gerar_mensagem_exemplo(usuario)
                info = client.publish(topico, json.dumps(mensagem), qos=1)
                info.wait_for_publish(timeout=5)
                print("Enviado (exemplo):", mensagem["timestamp"])
                time.sleep(2.0 / velocidade)
    except KeyboardInterrupt:
        print("\nSimulador interrompido.")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
