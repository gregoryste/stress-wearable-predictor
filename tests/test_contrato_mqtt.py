import json
from datetime import datetime

import numpy as np

from simulador.simulador_mqtt import (
    calcular_total_segundos,
    gerar_mensagem_exemplo,
    montar_payload_wesad,
)


def _validar_contrato_base(msg: dict, usuario_esperado: str):
    assert msg["usuario"] == usuario_esperado
    assert isinstance(msg["timestamp"], str)
    instante = datetime.fromisoformat(msg["timestamp"])
    assert instante.tzinfo is not None

    assert isinstance(msg["sinais"], dict)
    for chave in ("bvp", "eda", "temp", "acc"):
        assert chave in msg["sinais"]
        assert isinstance(msg["sinais"][chave], list)
        assert len(msg["sinais"][chave]) > 0

    assert isinstance(msg["contexto"], dict)
    assert isinstance(msg["contexto"]["horario"], str)
    assert isinstance(msg["contexto"]["ruido_db"], (int, float))

    # Garante que o payload é serializável em JSON (sem tipos numpy soltos)
    serializado = json.dumps(msg)
    assert isinstance(serializado, str)


def test_contrato_mensagem_exemplo():
    msg = gerar_mensagem_exemplo("S2")
    _validar_contrato_base(msg, "S2")


def test_contrato_payload_wesad_um_segundo():
    wrist_simulado = {
        "BVP": np.ones((128, 1), dtype=np.float64),
        "ACC": np.zeros((64, 3), dtype=np.float64),
        "EDA": np.full((8, 1), 1.5, dtype=np.float64),
        "TEMP": np.full((8, 1), 33.2, dtype=np.float64),
    }

    assert calcular_total_segundos(wrist_simulado) == 2

    msg = montar_payload_wesad("S2", wrist_simulado, segundo_idx=1)
    _validar_contrato_base(msg, "S2")

    assert len(msg["sinais"]["bvp"]) == 64
    assert len(msg["sinais"]["acc"]) == 32
    assert len(msg["sinais"]["acc"][0]) == 3
    assert len(msg["sinais"]["eda"]) == 4
    assert len(msg["sinais"]["temp"]) == 4
