"""Testes do treino. Precisam das dependências de treino (treino/requirements.txt);
onde elas não estão instaladas, o arquivo inteiro é pulado."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")

import joblib  # noqa: E402

from treino.treinar import (  # noqa: E402
    CALIBRACAO_S,
    FEATURES,
    estatisticas_calibracao,
    marcar_calibracao,
    normalizar,
    pares_proxima_janela,
    treinar,
)

BLOCOS = [("repouso", 600), ("estresse", 300), ("diversao", 300)]


def tabela_sintetica(participantes: int = 4, semente: int = 0) -> pd.DataFrame:
    """Janelas de 60 s a cada 10 s, em blocos de repouso, estresse e diversão.

    Cada pessoa tem um nível próprio (deslocamento) e o estresse soma um efeito
    fixo: só comparando com o repouso da própria pessoa o estado fica claro.
    """
    rng = np.random.default_rng(semente)
    efeito = {"repouso": 0.0, "estresse": 1.0, "diversao": 0.4}
    linhas = []
    for p in range(participantes):
        nome, deslocamento, inicio_bloco = f"S{p + 2}", rng.normal(0, 3), 0
        for estado, duracao in BLOCOS:
            for inicio in range(inicio_bloco, inicio_bloco + duracao - 60 + 1, 10):
                valores = {f: deslocamento + 5 * efeito[estado] + rng.normal(0, 0.3) for f in FEATURES}
                linhas.append({"participante": nome, "inicio_s": inicio, "fim_s": inicio + 60,
                               "rotulo": 0, "estado": estado, **valores})
            inicio_bloco += duracao + 60  # intervalo entre condições, como as transições do WESAD
    return pd.DataFrame(linhas)


def test_calibracao_sao_os_primeiros_minutos_de_repouso():
    df = tabela_sintetica()
    calibracao, sobreposta = marcar_calibracao(df)
    s2 = df[df.participante == "S2"]
    assert set(s2[calibracao].inicio_s) == set(range(0, CALIBRACAO_S - 60 + 1, 10))
    assert (s2[calibracao].estado == "repouso").all()
    # Janelas que começam antes do fim da calibração compartilham sinal com ela.
    assert set(s2[sobreposta].inicio_s) == set(range(CALIBRACAO_S - 50, CALIBRACAO_S, 10))
    assert not (calibracao & sobreposta).any()


def test_normalizar_zera_o_repouso_de_calibracao():
    df = tabela_sintetica()
    calibracao, _ = marcar_calibracao(df)
    normalizado = normalizar(df, estatisticas_calibracao(df, calibracao))
    medias = normalizado[calibracao].groupby(df.participante[calibracao]).mean()
    assert np.allclose(medias.to_numpy(), 0.0, atol=1e-9)


def test_pares_ligam_janelas_com_60_s_de_diferenca():
    df = tabela_sintetica(participantes=1)
    pares = pares_proxima_janela(df)
    assert len(pares) > 0
    assert (pares.inicio_seguinte - pares.inicio_s == 60).all()
    assert (pares.participante == "S2").all()


def test_treinar_gera_modelo_metadados_metricas_e_graficos(tmp_path):
    tabela = tmp_path / "features.csv"
    tabela_sintetica().to_csv(tabela, index=False)

    resultado = treinar(tabela, tmp_path / "modelos", tmp_path / "resultados", n_arvores=20)

    # O padrão pessoal resolve o deslocamento de cada pessoa; o modelo geral não.
    assert resultado["avaliacao"]["pessoal"]["f1_macro"] > 90
    assert resultado["avaliacao"]["pessoal"]["f1_macro"] > resultado["avaliacao"]["geral"]["f1_macro"]

    pacote = joblib.load(tmp_path / "modelos" / "modelo_estresse.joblib")
    assert pacote["features"] == FEATURES
    assert set(pacote["classes"]) == {"repouso", "estresse", "diversao"}
    assert set(pacote["calibracao"]) == {"S2", "S3", "S4", "S5"}
    linha = np.zeros((1, len(FEATURES)))
    assert pacote["pessoal"].predict(linha)[0] in pacote["classes"]

    metadados = json.loads((tmp_path / "modelos" / "metadados.json").read_text(encoding="utf-8"))
    assert metadados["versao"] == pacote["versao"]
    assert metadados["features"] == FEATURES
    assert "scikit-learn" in metadados["versoes"]

    metricas = json.loads((tmp_path / "resultados" / "metricas.json").read_text(encoding="utf-8"))
    assert metricas["validacao"].startswith("leave-one-subject-out")
    for nome in ("matriz_confusao.png", "f1_por_participante.png", "importancia_features.png"):
        assert (tmp_path / "resultados" / nome).stat().st_size > 0
