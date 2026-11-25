import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api_ia.main import app
from contratos.api import JanelaFeatures, Predicao

EXEMPLOS = Path(__file__).resolve().parent.parent / "contratos" / "exemplos"
client = TestClient(app)


def carregar(nome: str) -> dict:
    return json.loads((EXEMPLOS / nome).read_text(encoding="utf-8"))


def test_exemplos_respeitam_o_contrato():
    JanelaFeatures.model_validate(carregar("janela_features.json"))
    Predicao.model_validate(carregar("predicao.json"))


def test_saude():
    resposta = client.get("/saude")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_prever_responde_no_formato_do_contrato():
    entrada = carregar("janela_features.json")
    resposta = client.post("/prever", json=entrada)

    assert resposta.status_code == 200
    predicao = Predicao.model_validate(resposta.json())
    assert predicao.usuario == entrada["usuario"]
    assert predicao.fim_janela == JanelaFeatures.model_validate(entrada).fim_janela


def test_prever_rejeita_feature_faltando():
    entrada = carregar("janela_features.json")
    del entrada["features"]["hrv_rmssd"]
    resposta = client.post("/prever", json=entrada)
    assert resposta.status_code == 422


def test_prever_rejeita_campo_desconhecido():
    entrada = carregar("janela_features.json")
    entrada["features"]["campo_inventado"] = 1.0
    resposta = client.post("/prever", json=entrada)
    assert resposta.status_code == 422


def test_prever_rejeita_janela_sem_fuso_horario():
    entrada = carregar("janela_features.json")
    entrada["fim_janela"] = "2026-11-27T14:04:00"
    resposta = client.post("/prever", json=entrada)
    assert resposta.status_code == 422


def test_predicao_rejeita_janela_sem_fuso_horario():
    saida = carregar("predicao.json")
    saida["fim_janela"] = "2026-11-27T14:04:00"
    with pytest.raises(ValidationError):
        Predicao.model_validate(saida)


def test_predicao_rejeita_probabilidade_fora_de_0_a_1():
    saida = carregar("predicao.json")
    saida["probabilidade"] = 1.5
    with pytest.raises(ValidationError):
        Predicao.model_validate(saida)
