from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api_ia.main import app
from contratos.api import Features, JanelaFeatures, Predicao
from shared.features import (
    SINAIS_OBRIGATORIOS,
    amostras_esperadas,
    extrair_features,
    juntar_mensagens,
)
from simulador.simulador_mqtt import gerar_mensagem_exemplo


def janela_de_exemplo() -> dict:
    """Simula 60 s de mensagens do simulador (uma a cada 2 s)."""
    return juntar_mensagens(gerar_mensagem_exemplo("S2") for _ in range(30))


def test_extrair_features_devolve_o_formato_do_contrato():
    features = extrair_features(janela_de_exemplo())
    assert isinstance(features, Features)
    assert set(features.model_dump()) == set(Features.model_fields)


def test_juntar_mensagens_concatena_cada_sinal():
    mensagem = gerar_mensagem_exemplo("S2")
    janela = juntar_mensagens([mensagem, mensagem])
    for sinal in SINAIS_OBRIGATORIOS:
        assert len(janela[sinal]) == 2 * len(mensagem["sinais"][sinal])


def test_juntar_mensagens_recusa_mensagem_fora_do_contrato():
    with pytest.raises(ValueError):
        juntar_mensagens([{"usuario": "S2"}])


@pytest.mark.parametrize("mensagem", ["texto qualquer", 42, None, ["lista"]])
def test_juntar_mensagens_recusa_mensagem_que_nao_e_objeto(mensagem):
    with pytest.raises(ValueError):
        juntar_mensagens([mensagem])


@pytest.mark.parametrize("sinal", SINAIS_OBRIGATORIOS)
def test_juntar_mensagens_recusa_mensagem_sem_um_sinal(sinal):
    completa = gerar_mensagem_exemplo("S2")
    incompleta = gerar_mensagem_exemplo("S2")
    del incompleta["sinais"][sinal]
    with pytest.raises(ValueError):
        juntar_mensagens([completa, incompleta])


def test_juntar_mensagens_recusa_sinal_que_nao_e_lista():
    mensagem = gerar_mensagem_exemplo("S2")
    mensagem["sinais"]["eda"] = 1.42
    with pytest.raises(ValueError):
        juntar_mensagens([mensagem])


@pytest.mark.parametrize("sinal", SINAIS_OBRIGATORIOS)
def test_extrair_features_recusa_sinal_ausente(sinal):
    janela = janela_de_exemplo()
    del janela[sinal]
    with pytest.raises(ValueError):
        extrair_features(janela)


@pytest.mark.parametrize("sinal", SINAIS_OBRIGATORIOS)
def test_extrair_features_recusa_sinal_vazio(sinal):
    janela = janela_de_exemplo()
    janela[sinal] = []
    with pytest.raises(ValueError):
        extrair_features(janela)


def test_amostras_esperadas_numa_janela_de_60_s():
    assert amostras_esperadas("bvp") == 3840
    assert amostras_esperadas("eda") == 240
    assert amostras_esperadas("temp") == 240
    assert amostras_esperadas("acc") == 1920
    with pytest.raises(ValueError):
        amostras_esperadas("ecg")


def test_fluxo_da_ingestao_ate_a_api():
    """O caminho que a ingestão vai fazer: mensagens -> features -> POST /prever."""
    features = extrair_features(janela_de_exemplo())
    janela = JanelaFeatures(
        usuario="S2",
        fim_janela=datetime(2026, 11, 27, 14, 4, tzinfo=timezone.utc),
        features=features,
    )
    resposta = TestClient(app).post("/prever", json=janela.model_dump(mode="json"))
    assert resposta.status_code == 200
    Predicao.model_validate(resposta.json())
