"""Testes da comparação entre bases (treino/avaliar_bases.py) e das partes da simulação
pela API que não precisam do WESAD (treino/simular_api.py). Precisam das dependências de
treino; onde elas não estão, o arquivo é pulado."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from shared.features import FREQUENCIAS_HZ  # noqa: E402
from treino.avaliar_bases import avaliar, avaliar_entre_bases, carregar  # noqa: E402
from treino.preparar_wesad import FREQUENCIA_ROTULOS_HZ  # noqa: E402
from treino.simular_api import janelas_da_sessao, resumo_participante  # noqa: E402
from treino.treinar import FEATURES  # noqa: E402

EFEITO = {"repouso": 0.0, "estresse": 1.0, "diversao": 0.4, "exercicio": 2.5}


def linha(participante, estado, deslocamento, rng, **extra):
    valores = {f: deslocamento + 5 * EFEITO[estado] + rng.normal(0, 0.3) for f in FEATURES}
    return {"participante": participante, "estado": estado, **extra, **valores}


@pytest.fixture
def tabelas(tmp_path):
    """Tabelas pequenas no formato das duas bases, com estados bem separáveis."""
    rng = np.random.default_rng(1)
    wesad = []
    for p in ("S2", "S3", "S4"):
        deslocamento, inicio_bloco = rng.normal(0, 3), 0
        for estado, duracao in (("repouso", 600), ("estresse", 300), ("diversao", 300)):
            for inicio in range(inicio_bloco, inicio_bloco + duracao - 60 + 1, 10):
                wesad.append(linha(p, estado, deslocamento, rng, inicio_s=inicio, fim_s=inicio + 60, rotulo=0))
            inicio_bloco += duracao + 60
    physionet = []
    for p in ("pn-S01", "pn-S03", "pn-f01"):
        deslocamento = rng.normal(0, 3)
        for estado, sessao, n, calibracao in (("repouso", "STRESS", 6, True), ("repouso", "STRESS", 12, False),
                                              ("estresse", "STRESS", 12, False), ("exercicio", "AEROBIC", 20, False)):
            for _ in range(n):
                physionet.append(linha(p, estado, deslocamento, rng, base="physionet", sessao=sessao,
                                       fase=estado, calibracao=calibracao))
    caminho_wesad, caminho_pn = tmp_path / "wesad.csv", tmp_path / "physionet.csv"
    pd.DataFrame(wesad).to_csv(caminho_wesad, index=False)
    pd.DataFrame(physionet).to_csv(caminho_pn, index=False)
    return caminho_wesad, caminho_pn


def test_carregar_tira_a_calibracao_e_junta_as_bases(tabelas):
    dados = carregar(*tabelas)
    assert set(dados.base) == {"wesad", "physionet"}
    assert not dados.calibracao.astype(bool).any()
    assert set(dados[dados.base == "physionet"].estado) == {"repouso", "estresse", "exercicio"}


def test_avaliar_produz_todos_os_experimentos(tabelas):
    resultado = avaliar(carregar(*tabelas), n_arvores=15)

    assert resultado["janelas"]["participantes"] == {"wesad": 3, "physionet": 3}
    assert set(resultado["cruzado"]) == {"treino_wesad_teste_physionet", "treino_physionet_teste_wesad"}
    for medida in resultado["cruzado"].values():
        assert 0 <= medida["f1_estresse"] <= 100
    assert 0 <= resultado["exercicio_modelo_so_wesad"]["chamado_de_estresse_pct"] <= 100

    # Com estados separáveis, o modelo combinado reconhece o exercício.
    assert resultado["loso"]["combinado_exercicio"]["reconhecido_como_exercicio_pct"] > 80

    limiares = [item["limiar_exercicio"] for item in resultado["supressao_por_exercicio"]]
    assert limiares == [0.5, 0.7, 0.8, 0.9, "nunca suprime"]
    # Quanto mais alto o limiar, menos o exercício é suprimido.
    falsos = [item["exercicio_chamado_de_estresse_pct"] for item in resultado["supressao_por_exercicio"]]
    assert falsos == sorted(falsos)


@pytest.fixture
def tabela_stress_predict(tmp_path):
    """Tabela pequena no formato do Stress-Predict (calibração já marcada)."""
    rng = np.random.default_rng(2)
    linhas = []
    for p in ("sp-S02", "sp-S03", "sp-S04"):
        deslocamento = rng.normal(0, 3)
        for estado, fase, n, calibracao in (("repouso", "Linha de base", 6, True), ("repouso", "Final", 12, False),
                                            ("estresse", "Stroop", 12, False)):
            for _ in range(n):
                linhas.append(linha(p, estado, deslocamento, rng, base="stress_predict", sessao="STRESS",
                                    fase=fase, calibracao=calibracao))
    caminho = tmp_path / "stress_predict.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    return caminho


def test_carregar_aceita_a_terceira_base(tabelas, tabela_stress_predict):
    dados = carregar(*tabelas, tabela_stress_predict)
    assert set(dados.base) == {"wesad", "physionet", "stress_predict"}
    assert not dados.calibracao.astype(bool).any()
    assert dados[dados.base == "stress_predict"].participante.nunique() == 3


def test_avaliar_entre_bases_testa_cada_base_nunca_vista(tabelas, tabela_stress_predict):
    resultado = avaliar_entre_bases(carregar(*tabelas, tabela_stress_predict), n_arvores=15)

    bases = {"wesad", "physionet", "stress_predict"}
    assert resultado["participantes"] == {b: 3 for b in bases}
    assert "exercicio" not in resultado["janelas"]["physionet"]
    assert set(resultado["dentro_da_base"]) == bases
    assert set(resultado["todas_juntas_loso"]) == bases

    # Cada base é testada com cada combinação das outras duas: 3 bases x 3 combinações.
    testes = resultado["base_nunca_vista"]
    assert len(testes) == 9
    for medida in testes:
        assert medida["teste"] not in medida["treino"]
        assert 0 <= medida["f1_estresse"] <= 100
        # O exercício (nunca visto no treino) só é medido nos testes no PhysioNet.
        assert ("exercicio_chamado_de_estresse_pct" in medida) == (medida["teste"] == "physionet")
    # Com estados bem separáveis, o modelo de outra base acerta o estresse.
    assert min(m["f1_estresse"] for m in testes) > 80


def test_resumo_participante_conta_alertas_e_atraso():
    sessao = pd.DataFrame({
        "inicio_min": range(10),
        "condicao": ["repouso"] * 3 + ["outro"] + ["estresse"] * 3 + ["diversao"] * 2 + ["outro"],
        "estado": ["repouso", "estresse", "repouso", "estresse", "repouso", "estresse",
                   "estresse", "diversao", "diversao", "repouso"],
        "intervencao": ["nenhuma", "respiracao_guiada", "nenhuma", "respiracao_guiada", "nenhuma",
                        "respiracao_guiada", "respiracao_guiada", "nenhuma", "nenhuma", "nenhuma"],
        "descartada": [False] * 10,
    })
    resumo = resumo_participante(sessao)
    assert resumo["minutos_de_estresse"] == 3
    assert resumo["minutos_ate_primeiro_alerta"] == 1.0     # estresse começa no 4, 1º alerta no 5
    assert resumo["estresse_com_alerta_pct"] == pytest.approx(66.7)
    assert resumo["minutos_calmos"] == 5 and resumo["alertas_falsos"] == 1
    assert resumo["minutos_sem_rotulo"] == 2 and resumo["alertas_sem_rotulo"] == 1
    assert resumo["acerto_janelas_rotuladas"] == pytest.approx(100 * 6 / 8, abs=0.1)


def test_janelas_da_sessao_sao_seguidas_e_marcam_trechos_mistos():
    blocos = [(1, 120), (0, 30), (2, 150)]  # repouso, transição, estresse (rótulos do WESAD)
    duracao = sum(segundos for _, segundos in blocos)
    rotulos = np.concatenate([np.full(s * FREQUENCIA_ROTULOS_HZ, r) for r, s in blocos])
    wrist = {
        "BVP": np.zeros((duracao * FREQUENCIAS_HZ["bvp"], 1)),
        "EDA": np.zeros((duracao * FREQUENCIAS_HZ["eda"], 1)),
        "TEMP": np.zeros((duracao * FREQUENCIAS_HZ["temp"], 1)),
        "ACC": np.zeros((duracao * FREQUENCIAS_HZ["acc"], 3)),
    }
    janelas = list(janelas_da_sessao(wrist, rotulos))
    assert [inicio for inicio, _, _ in janelas] == [0, 60, 120, 180, 240]
    assert [condicao for _, condicao, _ in janelas] == ["repouso", "repouso", "outro", "estresse", "estresse"]
    assert len(janelas[0][2]["bvp"]) == 60 * FREQUENCIAS_HZ["bvp"]
