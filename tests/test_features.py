from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api_ia.main import app
from contratos.api import Features, JanelaFeatures, Predicao
from shared.features import (
    ACC_UNIDADES_POR_G,
    FREQUENCIAS_HZ,
    SINAIS_OBRIGATORIOS,
    amostras_esperadas,
    extrair_features,
    juntar_mensagens,
)
from simulador.simulador_mqtt import gerar_mensagem_exemplo

DURACAO_S = 60


def janela_sintetica(
    fc_bpm: float = 75.0,
    variacao_ibi_ms: float = 20.0,
    picos_eda: int = 3,
    eda_inclinacao_min: float = 0.1,
    temp_inclinacao_min: float = -0.2,
    tremor_g: float = 0.0,
) -> dict:
    """Janela de 60 s com valores conhecidos, no formato do simulador (unidades do WESAD).

    Os intervalos entre batimentos alternam entre (IBI - variação) e (IBI + variação):
    o RMSSD esperado é 2 x variação e o SDNN, cerca de 1 x variação.
    """
    # Pulso: um pulso gaussiano por batimento.
    fs = FREQUENCIAS_HZ["bvp"]
    t = np.arange(DURACAO_S * fs) / fs
    ibi_s = 60.0 / fc_bpm
    batimentos, instante, i = [], 0.3, 0
    while instante < DURACAO_S:
        batimentos.append(instante)
        instante += ibi_s + (variacao_ibi_ms / 1000.0) * (1 if i % 2 else -1)
        i += 1
    bvp = sum(100.0 * np.exp(-0.5 * ((t - b) / 0.08) ** 2) for b in batimentos)

    # Condutância da pele: tendência linear + respostas espaçadas (subida rápida, descida lenta).
    fs = FREQUENCIAS_HZ["eda"]
    t = np.arange(DURACAO_S * fs) / fs
    eda = 0.5 + eda_inclinacao_min * t / 60.0
    for inicio in np.linspace(8, 52, picos_eda) if picos_eda else []:
        dt = t - inicio
        eda = eda + np.where(dt >= 0, 0.15 * (1 - np.exp(-dt / 0.7)) * np.exp(-dt / 3.0), 0.0)

    temp = 33.0 + temp_inclinacao_min * t / 60.0

    # Acelerômetro: gravidade no eixo z, mais um tremor opcional no eixo x.
    fs = FREQUENCIAS_HZ["acc"]
    t = np.arange(DURACAO_S * fs) / fs
    acc = np.zeros((len(t), 3))
    acc[:, 2] = 1.0
    acc[:, 0] = tremor_g * np.sin(2 * np.pi * 3 * t)
    acc *= ACC_UNIDADES_POR_G

    return {"bvp": bvp.tolist(), "eda": eda.tolist(), "temp": temp.tolist(), "acc": acc.tolist()}


def janela_em_mensagens(janela: dict) -> list[dict]:
    """Divide a janela em mensagens de 1 s, como o simulador publica."""
    mensagens = []
    for s in range(DURACAO_S):
        sinais = {
            nome: janela[nome][s * FREQUENCIAS_HZ[nome] : (s + 1) * FREQUENCIAS_HZ[nome]]
            for nome in SINAIS_OBRIGATORIOS
        }
        mensagens.append({"usuario": "S2", "timestamp": f"2026-11-27T14:03:{s:02d}Z", "sinais": sinais})
    return mensagens


# --- formato -------------------------------------------------------------------


def test_extrair_features_devolve_o_formato_do_contrato():
    features = extrair_features(janela_sintetica())
    assert isinstance(features, Features)
    assert set(features.model_dump()) == set(Features.model_fields)


# --- valores calculados ----------------------------------------------------------


@pytest.mark.parametrize("fc_bpm", [60.0, 75.0, 110.0])
def test_frequencia_cardiaca(fc_bpm):
    features = extrair_features(janela_sintetica(fc_bpm=fc_bpm))
    assert features.fc_media == pytest.approx(fc_bpm, abs=2.0)


def test_variabilidade_cardiaca():
    features = extrair_features(janela_sintetica(fc_bpm=75.0, variacao_ibi_ms=20.0))
    assert features.hrv_rmssd == pytest.approx(40.0, abs=12.0)
    assert features.hrv_sdnn == pytest.approx(20.0, abs=8.0)


def test_variabilidade_maior_quando_os_batimentos_variam_mais():
    pouca = extrair_features(janela_sintetica(variacao_ibi_ms=10.0))
    muita = extrair_features(janela_sintetica(variacao_ibi_ms=40.0))
    assert muita.hrv_rmssd > pouca.hrv_rmssd


@pytest.mark.parametrize("picos", [0, 3, 6])
def test_picos_de_condutancia(picos):
    assert extrair_features(janela_sintetica(picos_eda=picos)).eda_picos == picos


def test_inclinacao_da_condutancia():
    subindo = extrair_features(janela_sintetica(picos_eda=0, eda_inclinacao_min=0.1))
    descendo = extrair_features(janela_sintetica(picos_eda=0, eda_inclinacao_min=-0.1))
    assert subindo.eda_inclinacao == pytest.approx(0.1, abs=0.01)
    assert descendo.eda_inclinacao == pytest.approx(-0.1, abs=0.01)


def test_temperatura():
    features = extrair_features(janela_sintetica(temp_inclinacao_min=-0.2))
    assert features.temp_inclinacao == pytest.approx(-0.2, abs=0.01)
    assert features.temp_media == pytest.approx(32.9, abs=0.01)


def test_movimento_desconta_a_gravidade():
    parado = extrair_features(janela_sintetica(tremor_g=0.0))
    tremendo = extrair_features(janela_sintetica(tremor_g=0.5))
    assert parado.movimento_media == pytest.approx(0.0, abs=1e-6)
    assert tremendo.movimento_media > 0.05


# --- janelas inválidas -----------------------------------------------------------


@pytest.mark.parametrize("sinal", SINAIS_OBRIGATORIOS)
def test_extrair_features_recusa_sinal_ausente(sinal):
    janela = janela_sintetica()
    del janela[sinal]
    with pytest.raises(ValueError):
        extrair_features(janela)


@pytest.mark.parametrize("sinal", SINAIS_OBRIGATORIOS)
def test_extrair_features_recusa_sinal_vazio(sinal):
    janela = janela_sintetica()
    janela[sinal] = []
    with pytest.raises(ValueError):
        extrair_features(janela)


@pytest.mark.parametrize("sinal", SINAIS_OBRIGATORIOS)
def test_extrair_features_recusa_janela_incompleta(sinal):
    janela = janela_sintetica()
    janela[sinal] = janela[sinal][: amostras_esperadas(sinal) // 3]
    with pytest.raises(ValueError, match="incompleta"):
        extrair_features(janela)


def test_extrair_features_aceita_janela_parcial_acima_do_minimo():
    janela = janela_sintetica()
    for sinal in SINAIS_OBRIGATORIOS:
        janela[sinal] = janela[sinal][: int(0.8 * amostras_esperadas(sinal))]
    assert extrair_features(janela).fc_media == pytest.approx(75.0, abs=2.0)


def test_extrair_features_recusa_mensagens_de_exemplo_curtas_demais():
    """As mensagens falsas do simulador têm só 2 a 3 amostras: não formam uma janela real."""
    janela = juntar_mensagens(gerar_mensagem_exemplo("S2") for _ in range(30))
    with pytest.raises(ValueError, match="incompleta"):
        extrair_features(janela)


def test_extrair_features_recusa_pulso_sem_batimentos():
    janela = janela_sintetica()
    janela["bvp"] = [0.0] * amostras_esperadas("bvp")
    with pytest.raises(ValueError, match="Batimentos insuficientes"):
        extrair_features(janela)


def test_extrair_features_recusa_valor_invalido():
    janela = janela_sintetica()
    janela["eda"][10] = float("nan")
    with pytest.raises(ValueError, match="inválidos"):
        extrair_features(janela)


def test_extrair_features_recusa_acelerometro_fora_do_formato():
    janela = janela_sintetica()
    janela["acc"] = [1.0] * amostras_esperadas("acc")
    with pytest.raises(ValueError, match="triplas"):
        extrair_features(janela)


# --- juntar mensagens ------------------------------------------------------------


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


def test_amostras_esperadas_numa_janela_de_60_s():
    assert amostras_esperadas("bvp") == 3840
    assert amostras_esperadas("eda") == 240
    assert amostras_esperadas("temp") == 240
    assert amostras_esperadas("acc") == 1920
    with pytest.raises(ValueError):
        amostras_esperadas("ecg")


# --- caminho completo ------------------------------------------------------------


def test_fluxo_da_ingestao_ate_a_api():
    """O caminho que a ingestão faz: 60 mensagens -> features -> POST /prever."""
    janela = juntar_mensagens(janela_em_mensagens(janela_sintetica()))
    features = extrair_features(janela)
    entrada = JanelaFeatures(
        usuario="S2",
        fim_janela=datetime(2026, 11, 27, 14, 4, tzinfo=timezone.utc),
        features=features,
    )
    resposta = TestClient(app).post("/prever", json=entrada.model_dump(mode="json"))
    assert resposta.status_code == 200
    Predicao.model_validate(resposta.json())


# --- dados reais (só roda onde o WESAD estiver baixado; o CI pula) -------------------

PKL_S2 = Path(__file__).resolve().parent.parent / "WESAD" / "S2" / "S2.pkl"


@pytest.mark.skipif(not PKL_S2.exists(), reason="WESAD não encontrado (pasta WESAD/ na raiz)")
def test_wesad_estresse_aumenta_frequencia_cardiaca_e_condutancia():
    from simulador.simulador_mqtt import carregar_sinais_wrist

    import pickle

    with open(PKL_S2, "rb") as arquivo:
        rotulos = pickle.load(arquivo, encoding="latin1")["label"]
    wrist = carregar_sinais_wrist(str(PKL_S2))

    def janela_no_meio_da_condicao(condicao: int) -> dict:
        segundos = np.flatnonzero(rotulos == condicao) / 700
        inicio = int(np.median(segundos)) - DURACAO_S // 2
        return {
            "bvp": wrist["BVP"][inicio * 64 : (inicio + DURACAO_S) * 64],
            "eda": wrist["EDA"][inicio * 4 : (inicio + DURACAO_S) * 4],
            "temp": wrist["TEMP"][inicio * 4 : (inicio + DURACAO_S) * 4],
            "acc": wrist["ACC"][inicio * 32 : (inicio + DURACAO_S) * 32],
        }

    repouso = extrair_features(janela_no_meio_da_condicao(1))
    estresse = extrair_features(janela_no_meio_da_condicao(2))
    for features in (repouso, estresse):
        assert 40 <= features.fc_media <= 200
    assert estresse.fc_media > repouso.fc_media
    assert estresse.eda_media > repouso.eda_media
