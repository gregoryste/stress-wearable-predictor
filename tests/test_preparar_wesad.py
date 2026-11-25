import csv
import json
import pickle

import numpy as np
import pytest

from shared.features import DURACAO_JANELA_S, FREQUENCIAS_HZ
from treino.preparar_wesad import (
    COLUNAS_FEATURES,
    COLUNAS_ID,
    FREQUENCIA_ROTULOS_HZ,
    janelas_do_participante,
    preparar,
)


def participante_sintetico(blocos: list[tuple[int, int]]) -> tuple[dict, np.ndarray]:
    """Sinais de pulso plausíveis e rótulos em blocos [(rótulo, duração em s), ...]."""
    duracao = sum(segundos for _, segundos in blocos)
    rotulos = np.concatenate(
        [np.full(segundos * FREQUENCIA_ROTULOS_HZ, rotulo) for rotulo, segundos in blocos]
    )
    t = np.arange(duracao * FREQUENCIAS_HZ["bvp"]) / FREQUENCIAS_HZ["bvp"]
    wrist = {
        "BVP": (100 * np.sin(2 * np.pi * 1.2 * t)).reshape(-1, 1),
        "EDA": np.full((duracao * FREQUENCIAS_HZ["eda"], 1), 0.4),
        "TEMP": np.full((duracao * FREQUENCIAS_HZ["temp"], 1), 33.0),
        "ACC": np.tile([0.0, 0.0, 64.0], (duracao * FREQUENCIAS_HZ["acc"], 1)),
    }
    return wrist, rotulos


def test_so_gera_janelas_inteiras_dentro_de_uma_condicao_valida():
    # 90 s de transição (0), 120 s de repouso (1), 30 s de meditação (4), 90 s de estresse (2)
    wrist, rotulos = participante_sintetico([(0, 90), (1, 120), (4, 30), (2, 90)])
    janelas = list(janelas_do_participante(wrist, rotulos, passo_s=30))
    assert [(inicio, rotulo) for inicio, rotulo, _ in janelas] == [
        (90, 1), (120, 1), (150, 1),   # repouso: 90-210 s
        (240, 2), (270, 2),            # estresse: 240-330 s
    ]


def test_cada_janela_tem_o_tamanho_certo_de_cada_sinal():
    wrist, rotulos = participante_sintetico([(1, 120)])
    for _, _, sinais in janelas_do_participante(wrist, rotulos, passo_s=60):
        for sinal, fs in FREQUENCIAS_HZ.items():
            assert len(sinais[sinal]) == fs * DURACAO_JANELA_S
        assert np.asarray(sinais["acc"]).shape == (FREQUENCIAS_HZ["acc"] * DURACAO_JANELA_S, 3)


def test_passo_menor_gera_mais_janelas():
    wrist, rotulos = participante_sintetico([(1, 300)])
    assert len(list(janelas_do_participante(wrist, rotulos, passo_s=10))) == 25
    assert len(list(janelas_do_participante(wrist, rotulos, passo_s=60))) == 5


def test_preparar_grava_tabela_e_resumo(tmp_path):
    pasta = tmp_path / "WESAD"
    for nome, blocos in {"S2": [(1, 120), (2, 120)], "S10": [(3, 90)]}.items():
        wrist, rotulos = participante_sintetico(blocos)
        (pasta / nome).mkdir(parents=True)
        with open(pasta / nome / f"{nome}.pkl", "wb") as arquivo:
            pickle.dump({"subject": nome, "signal": {"wrist": wrist}, "label": rotulos}, arquivo)

    saida = tmp_path / "data" / "features.csv"
    resumo = preparar(pasta, saida, passo_s=30)

    with open(saida, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert list(linhas[0]) == COLUNAS_ID + COLUNAS_FEATURES
    assert [linha["participante"] for linha in linhas] == ["S2"] * 6 + ["S10"] * 2
    assert {linha["estado"] for linha in linhas} == {"repouso", "estresse", "diversao"}
    assert float(linhas[0]["fc_media"]) == pytest.approx(72.0, abs=2.0)

    assert resumo["janelas"] == 8
    assert resumo["janelas_por_estado"] == {"repouso": 3, "estresse": 3, "diversao": 2}
    gravado = json.loads(saida.with_suffix(".resumo.json").read_text(encoding="utf-8"))
    assert gravado["janelas"] == 8


def test_preparar_avisa_quando_nao_acha_o_wesad(tmp_path):
    with pytest.raises(FileNotFoundError):
        preparar(tmp_path / "nao-existe", tmp_path / "saida.csv", passo_s=10)
