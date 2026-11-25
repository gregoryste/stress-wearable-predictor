"""Testes do leitor da base do PhysioNet, com pastas sintéticas no formato da Empatica E4.
Precisam do pandas (dependências de treino); onde ele não está, o arquivo é pulado."""

import csv

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from treino.preparar_physionet import fases_da_sessao, ler_sinal, ler_toques, preparar  # noqa: E402

INICIO = pd.Timestamp("2013-02-20 17:00:00")

# Toques (s desde o início) escolhidos para cada fase ter um número conhecido de janelas.
TOQUES_V1 = [10, 130, 150, 250, 400, 520, 640, 680, 690, 720, 730, 770, 780]
TOQUES_V2 = [10, 100, 300, 400, 560, 580, 700, 820, 860]
TOQUES_EXERCICIO = [60, 300, 600]


def escrever_sessao(pasta, duracao_s: int, toques: list[int] | None) -> None:
    """Cria ACC/BVP/EDA/TEMP/tags.csv como a E4: início, frequência e valores."""
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
            arquivo.write(",".join([str(INICIO)] * colunas) + "\n")
            arquivo.write(",".join([f"{frequencia:.1f}"] * colunas) + "\n")
            np.savetxt(arquivo, valores, delimiter=",", fmt="%.4f")
    with open(pasta / "tags.csv", "w", encoding="utf-8") as arquivo:
        for segundo in toques or []:
            arquivo.write(f"{INICIO + pd.Timedelta(seconds=segundo)}\n")


@pytest.fixture
def base(tmp_path):
    raiz = tmp_path / "Wearable_Dataset"
    escrever_sessao(raiz / "STRESS" / "S01", 900, TOQUES_V1)
    escrever_sessao(raiz / "STRESS" / "f01", 900, TOQUES_V2)
    escrever_sessao(raiz / "STRESS" / "S02", 900, TOQUES_V1)          # excluído (dados duplicados)
    escrever_sessao(raiz / "AEROBIC" / "S01", 700, TOQUES_EXERCICIO)
    escrever_sessao(raiz / "AEROBIC" / "S11_a", 700, TOQUES_EXERCICIO)  # sessão dividida: fica fora
    escrever_sessao(raiz / "ANAEROBIC" / "S01", 700, [])              # sem toques: sem rótulo
    return raiz


def test_ler_sinal_le_inicio_frequencia_e_valores(base):
    inicio, frequencia, valores = ler_sinal(base / "STRESS" / "S01" / "ACC.csv")
    assert inicio == INICIO
    assert frequencia == 32.0
    assert valores.shape == (900 * 32, 3)
    _, _, bvp = ler_sinal(base / "STRESS" / "S01" / "BVP.csv")
    assert bvp.shape == (900 * 64, 1)


def test_ler_toques_em_segundos_desde_o_inicio(base):
    assert ler_toques(base / "STRESS" / "S01", INICIO).tolist() == TOQUES_V1
    assert len(ler_toques(base / "ANAEROBIC" / "S01", INICIO)) == 0


def test_fases_da_versao_1():
    fases = fases_da_sessao("STRESS", "S01", np.array(TOQUES_V1, dtype=float))
    assert [(nome, a, b, estado) for nome, a, b, estado in fases] == [
        ("Baseline", 10, 130, "repouso"),
        ("Stroop", 150, 250, "estresse"),
        ("First Rest", 250, 400, "repouso"),
        ("TMCT", 400, 520, "estresse"),
        ("Second Rest", 520, 640, "repouso"),
        ("Opinioes e contas", 640, 770, "estresse"),
    ]


def test_fases_da_versao_2():
    fases = fases_da_sessao("STRESS", "f01", np.array(TOQUES_V2, dtype=float))
    assert [(nome, a, b, estado) for nome, a, b, estado in fases] == [
        ("Baseline", 10, 100, "repouso"),
        ("TMCT", 100, 300, "estresse"),
        ("First Rest", 300, 400, "repouso"),
        ("Opinioes", 400, 700, "estresse"),
        ("Second Rest", 700, 820, "repouso"),
    ]


def test_exercicio_vai_do_primeiro_ao_ultimo_toque():
    assert fases_da_sessao("AEROBIC", "S01", np.array(TOQUES_EXERCICIO, dtype=float)) == [
        ("Aerobic", 60, 600, "exercicio")
    ]
    assert fases_da_sessao("ANAEROBIC", "S01", np.empty(0)) == []


def test_preparar_rotula_janelas_e_pula_os_excluidos(base, tmp_path):
    saida = tmp_path / "features_physionet.csv"
    resumo = preparar(base, saida)

    with open(saida, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    tabela = pd.DataFrame(linhas)
    contagem = tabela.groupby(["participante", "estado"]).size().to_dict()

    # Janelas de 60 s a cada 10 s, inteiras dentro de cada fase.
    assert contagem == {
        ("pn-S01", "repouso"): 7 + 10 + 7,     # linha de base, 1º e 2º descanso
        ("pn-S01", "estresse"): 5 + 7 + 8,     # Stroop, TMCT, opiniões e contas
        ("pn-S01", "exercicio"): 49,           # aeróbico, de 60 a 600 s
        ("pn-f01", "repouso"): 4 + 5 + 7,
        ("pn-f01", "estresse"): 15 + 25,
    }
    calibracao = tabela[tabela.calibracao == "True"]
    assert set(calibracao.fase) == {"Baseline"}
    assert calibracao.groupby("participante").size().to_dict() == {"pn-S01": 7, "pn-f01": 4}
    assert set(tabela.base) == {"physionet"}
    assert float(tabela.fc_media.astype(float).median()) == pytest.approx(72.0, abs=2.0)

    assert "STRESS/S02" in resumo["pulados"]
    assert "AEROBIC/S11_a" in resumo["pulados"]
    assert resumo["janelas_por_estado"] == {"repouso": 40, "estresse": 60, "exercicio": 49}
