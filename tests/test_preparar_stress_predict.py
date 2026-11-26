"""Testes do leitor da base Stress-Predict, com pastas sintéticas no formato da Empatica E4.
Precisam do pandas (dependências de treino); onde ele não está, o arquivo é pulado."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from treino.preparar_stress_predict import (  # noqa: E402
    ARQUIVO_ROTULOS,
    fases_do_participante,
    ler_sinal,
    preparar,
    trechos_rotulados,
)

INICIO = 1644227574.0
DURACAO_S = 1500
# (início, fim, rótulo) em segundos desde o início da gravação, na ordem do protocolo.
TRECHOS_S02 = [(10, 500, 0), (500, 700, 1), (700, 900, 0), (900, 1200, 1), (1200, 1300, 0),
               (1300, 1420, 1), (1420, 1500, 0)]


def escrever_participante(pasta, duracao_s: int = DURACAO_S) -> None:
    """Cria ACC/BVP/EDA/TEMP.csv como a E4: início (Unix), frequência e valores."""
    pasta.mkdir(parents=True)
    t = np.arange(duracao_s * 64) / 64
    sinais = {
        "BVP.csv": (64, (100 * np.sin(2 * np.pi * 1.2 * t)).reshape(-1, 1)),
        "EDA.csv": (4, np.full((duracao_s * 4, 1), 0.4)),
        "TEMP.csv": (4, np.full((duracao_s * 4, 1), 33.0)),
        "ACC.csv": (32, np.tile([0.0, 0.0, 64.0], (duracao_s * 32, 1))),
    }
    for nome, (frequencia, valores) in sinais.items():
        colunas = valores.shape[1]
        with open(pasta / nome, "w", encoding="utf-8") as arquivo:
            arquivo.write(", ".join([f"{INICIO:.6f}"] * colunas) + "\n")
            arquivo.write(", ".join([f"{frequencia:.6f}"] * colunas) + "\n")
            np.savetxt(arquivo, valores, delimiter=",", fmt="%.4f")


def linhas_de_rotulo(participante: int, trechos) -> list[dict]:
    """Uma linha por segundo, como o arquivo processado dos autores."""
    return [
        {"Participant": participante, "HR": 72, "respr": 12.0, "Time(sec)": int(INICIO) + s, "Label": rotulo}
        for a, b, rotulo in trechos
        for s in range(a, b)
    ]


@pytest.fixture
def base(tmp_path):
    raiz = tmp_path / "repo"
    for participante in ("S01", "S02", "S03"):
        escrever_participante(raiz / "Raw_data" / participante)
    (raiz / "Raw_data" / "Readme").write_text("arquivo solto, não é participante\n")
    linhas = linhas_de_rotulo(2, TRECHOS_S02) + linhas_de_rotulo(3, [(10, 400, 0), (400, 600, 1), (600, 900, 0)])
    (raiz / ARQUIVO_ROTULOS).parent.mkdir(parents=True)
    pd.DataFrame(linhas).sample(frac=1, random_state=0).to_csv(raiz / ARQUIVO_ROTULOS, index=False)  # fora de ordem
    return raiz


def test_ler_sinal_le_inicio_unix_frequencia_e_valores(base):
    inicio, frequencia, valores = ler_sinal(base / "Raw_data" / "S02" / "ACC.csv")
    assert inicio == INICIO
    assert frequencia == 32.0
    assert valores.shape == (DURACAO_S * 32, 3)


def test_trechos_rotulados_juntam_segundos_seguidos_mesmo_fora_de_ordem():
    segundos = np.array([3, 0, 1, 2, 4, 5], dtype=float)
    rotulos = np.array([1, 0, 0, 1, 1, 0])
    assert trechos_rotulados(segundos, rotulos) == [(0, 2, 0), (2, 5, 1), (5, 6, 0)]


def test_fases_separam_a_calibracao_antes_do_stroop():
    fases = fases_do_participante(TRECHOS_S02)
    assert fases == [
        ("Inicio", 10, 200, "repouso"),
        ("Linha de base", 200, 500, "repouso"),
        ("Stroop", 500, 700, "estresse"),
        ("Descanso 1", 700, 900, "repouso"),
        ("Entrevista", 900, 1200, "estresse"),
        ("Descanso 2", 1200, 1300, "repouso"),
        ("Hiperventilacao", 1300, 1420, "estresse"),
        ("Final", 1420, 1500, "repouso"),
    ]


def test_inicio_curto_vira_todo_calibracao():
    trechos = [(0, 200, 0)] + TRECHOS_S02[1:]
    fases = fases_do_participante(trechos)
    assert fases[0] == ("Linha de base", 0, 200, "repouso")
    assert fases[1][0] == "Stroop"


def test_trechos_fora_do_protocolo_levantam_erro():
    with pytest.raises(ValueError, match="fora do protocolo"):
        fases_do_participante([(0, 300, 0), (300, 600, 1), (600, 900, 0)])


def test_preparar_rotula_janelas_e_pula_quem_nao_tem_rotulo(base, tmp_path):
    saida = tmp_path / "features_stress_predict.csv"
    resumo = preparar(base, saida)
    tabela = pd.read_csv(saida)

    # Janelas de 60 s a cada 10 s, inteiras dentro de cada fase.
    assert tabela.groupby("fase").size().to_dict() == {
        "Inicio": 14, "Linha de base": 25, "Stroop": 15, "Descanso 1": 15,
        "Entrevista": 25, "Descanso 2": 5, "Hiperventilacao": 7, "Final": 3,
    }
    assert set(tabela.participante) == {"sp-S02"}
    assert set(tabela.base) == {"stress_predict"}
    assert set(tabela[tabela.calibracao].fase) == {"Linha de base"}
    assert float(tabela.fc_media.median()) == pytest.approx(72.0, abs=2.0)

    assert resumo["janelas_por_estado"] == {"repouso": 14 + 25 + 15 + 5 + 3, "estresse": 15 + 25 + 7}
    assert "S01 (sem rótulos)" in resumo["pulados"]
    assert any(item.startswith("S03 (trechos fora do protocolo") for item in resumo["pulados"])
