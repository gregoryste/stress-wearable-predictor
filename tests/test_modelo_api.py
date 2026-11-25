import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api_ia.main import app
from api_ia.modelo import ServicoModelo
from api_ia.regras import (
    INTERVALO_NOVA_SESSAO,
    JANELAS_ESTRESSE_LONGO,
    Historico,
    Registro,
    escolher_intervencao,
    prever_proxima,
    projetar_estresse,
)
from contratos.api import Intervencao, JanelaFeatures, Predicao

METADADOS = Path(__file__).resolve().parent.parent / "modelos" / "metadados.json"
INICIO = datetime(2026, 11, 27, 14, 0, tzinfo=timezone.utc)

# Diferença típica entre estresse e repouso no WESAD (medianas das 3039 janelas).
EFEITO_ESTRESSE = {"fc_media": 23.3, "fc_desvio": 4.1, "eda_media": 1.8, "eda_picos": 7,
                   "eda_inclinacao": 0.03, "temp_media": -0.8, "temp_inclinacao": -0.09}


@pytest.fixture
def servico() -> ServicoModelo:
    return ServicoModelo()  # novo a cada teste: histórico vazio


def janela(servico: ServicoModelo, usuario: str = "S2", minuto: int = 0, estresse: float = 0.0) -> JanelaFeatures:
    """Janela com as features do repouso de calibração de S2, somadas a uma fração do efeito de estresse."""
    repouso = dict(zip(servico.features, servico._repouso["S2"]))
    valores = {f: repouso[f] + estresse * EFEITO_ESTRESSE.get(f, 0.0) for f in servico.features}
    valores["eda_picos"] = max(0, round(valores["eda_picos"]))
    valores["eda_media"] = max(0.0, valores["eda_media"])
    return JanelaFeatures(usuario=usuario, fim_janela=INICIO + timedelta(minutes=minuto), features=valores)


# --- modelo real --------------------------------------------------------------------


def test_versao_do_modelo_carregado_e_a_dos_metadados(servico):
    assert servico.versao == json.loads(METADADOS.read_text(encoding="utf-8"))["versao"]


def test_usuario_calibrado_usa_modelo_pessoal_e_desconhecido_usa_o_geral(servico):
    assert "S2" in servico.usuarios_calibrados
    assert servico.variante("S2") == "pessoal"
    assert servico.variante("usuario-novo") == "geral"


def test_probabilidades_somam_1(servico):
    probabilidades = servico.probabilidades(janela(servico))
    assert set(probabilidades) == {"repouso", "estresse", "diversao"}
    assert sum(probabilidades.values()) == pytest.approx(1.0)


def test_repouso_da_propria_pessoa_e_repouso(servico):
    predicao = servico.prever(janela(servico, estresse=0.0))
    assert predicao.estado.value == "repouso"
    assert predicao.intervencao == Intervencao.nenhuma


def test_estresse_tipico_e_confirmado_na_segunda_janela_seguida(servico):
    primeira = servico.prever(janela(servico, minuto=0, estresse=1.0))
    assert primeira.estado.value == "estresse"
    assert primeira.probabilidade > 0.5
    assert primeira.intervencao == Intervencao.nenhuma          # aguarda confirmação
    segunda = servico.prever(janela(servico, minuto=1, estresse=1.0))
    assert segunda.intervencao == Intervencao.respiracao_guiada


def test_usuario_sem_calibracao_tambem_recebe_predicao_valida(servico):
    predicao = servico.prever(janela(servico, usuario="usuario-novo"))
    Predicao.model_validate(predicao.model_dump())


# --- API -------------------------------------------------------------------------------


def test_api_saude_mostra_o_modelo_real():
    resposta = TestClient(app).get("/saude")
    assert resposta.status_code == 200
    assert resposta.json()["versao_modelo"].startswith("rf-")
    assert resposta.json()["usuarios_calibrados"] == 15


def test_api_prever_responde_com_o_modelo_real():
    entrada = janela(ServicoModelo(), estresse=1.0).model_dump(mode="json")
    resposta = TestClient(app).post("/prever", json=entrada)
    assert resposta.status_code == 200
    predicao = Predicao.model_validate(resposta.json())
    assert predicao.versao_modelo.startswith("rf-")
    assert predicao.estado.value == "estresse"


# --- regras de tendência e intervenção ----------------------------------------------------


def registros(*probabilidades: float) -> list[Registro]:
    return [
        Registro(INICIO + timedelta(minutes=i), "estresse" if p >= 0.5 else "repouso", p)
        for i, p in enumerate(probabilidades)
    ]


def test_tendencia_de_subida_projeta_a_proxima_janela():
    assert projetar_estresse(registros(0.1, 0.25, 0.4)) == pytest.approx(0.55)
    assert projetar_estresse(registros(0.4)) == pytest.approx(0.4)
    assert projetar_estresse(registros(0.7, 0.9, 1.0)) == 1.0


def test_subida_sem_estresse_ainda_gera_aviso_antecipado():
    historico = registros(0.1, 0.25, 0.4)
    proxima, confianca = prever_proxima(historico, {"repouso": 0.55, "estresse": 0.4, "diversao": 0.05})
    assert proxima == "estresse"
    assert confianca == pytest.approx(0.55)
    assert escolher_intervencao(historico, proxima) == Intervencao.aviso_antecipado


def test_estavel_sem_estresse_nao_gera_intervencao():
    historico = registros(0.1, 0.1, 0.1)
    proxima, confianca = prever_proxima(historico, {"repouso": 0.6, "estresse": 0.1, "diversao": 0.3})
    assert proxima == "repouso"
    assert confianca == pytest.approx(0.9)
    assert escolher_intervencao(historico, proxima) == Intervencao.nenhuma


def test_pico_isolado_de_estresse_nao_gera_alerta():
    pico = registros(0.1, 0.1, 0.9)
    assert escolher_intervencao(pico, "estresse") == Intervencao.nenhuma
    depois_do_pico = registros(0.1, 0.9, 0.1)
    assert escolher_intervencao(depois_do_pico, "repouso") == Intervencao.nenhuma


def test_estresse_curto_respiracao_e_longo_pausa_ativa():
    curto = registros(*([0.1] * 3 + [0.8] * 2))
    longo = registros(*([0.8] * JANELAS_ESTRESSE_LONGO))
    assert escolher_intervencao(curto, "estresse") == Intervencao.respiracao_guiada
    assert escolher_intervencao(longo, "estresse") == Intervencao.pausa_ativa


def test_historico_nao_duplica_janela_reenviada():
    historico = Historico()
    historico.registrar("S2", Registro(INICIO, "repouso", 0.1))
    atual = historico.registrar("S2", Registro(INICIO, "estresse", 0.9))
    assert len(atual) == 1
    assert atual[0].prob_estresse == 0.9


def test_historico_recomeca_apos_pausa_longa_ou_janela_fora_de_ordem():
    historico = Historico()
    historico.registrar("S2", Registro(INICIO, "repouso", 0.1))
    historico.registrar("S2", Registro(INICIO + timedelta(minutes=1), "repouso", 0.1))
    depois_da_pausa = INICIO + timedelta(minutes=1) + INTERVALO_NOVA_SESSAO + timedelta(seconds=1)
    assert len(historico.registrar("S2", Registro(depois_da_pausa, "repouso", 0.1))) == 1
    assert len(historico.registrar("S2", Registro(INICIO, "repouso", 0.1))) == 1


def test_historico_separa_usuarios():
    historico = Historico()
    historico.registrar("S2", Registro(INICIO, "repouso", 0.1))
    assert len(historico.registrar("S3", Registro(INICIO, "repouso", 0.1))) == 1


def test_servico_completo_avisa_antes_e_depois_sugere_respiracao(servico):
    """Estresse subindo minuto a minuto: primeiro nenhuma, depois aviso, depois respiração."""
    intervencoes = [
        servico.prever(janela(servico, minuto=m, estresse=nivel)).intervencao
        for m, nivel in enumerate([0.0, 0.0, 0.0, 0.5, 0.7, 1.0, 1.0])
    ]
    assert intervencoes[0] == Intervencao.nenhuma
    assert intervencoes[-1] == Intervencao.respiracao_guiada
    primeira_resposta = next(i for i in intervencoes if i != Intervencao.nenhuma)
    assert primeira_resposta in (Intervencao.aviso_antecipado, Intervencao.respiracao_guiada)
