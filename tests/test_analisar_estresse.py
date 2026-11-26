"""Testes da análise "o que é estresse" (treino/analisar_estresse.py): leitura dos
questionários, junção com as janelas e medidas. Precisam das dependências de treino;
onde elas não estão, o arquivo é pulado."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from treino.analisar_estresse import (  # noqa: E402
    analisar,
    correlacao_dentro_da_pessoa,
    ler_relatos_physionet,
    ler_relatos_wesad,
    por_fase,
)


def escrever_quest(pasta, participante: str, ordem: list[str], estressado: dict[str, int]) -> None:
    """SX_quest.csv no formato do WESAD, com o item "Stressed" do PANAS e o STAI escolhidos."""
    pasta = pasta / participante
    pasta.mkdir(parents=True)
    linhas = [f"# Subj;{participante};;;", "# ORDER;" + ";".join(ordem + ["bRead", "sRead"]) + ";;",
              "# START;1;2;3;4;5;6;7;;", "# END;1;2;3;4;5;6;7;;", ";;;;"]
    condicoes = [c for c in ordem if not c.endswith("Read")]
    for condicao in condicoes:
        panas = [1] * 24 + ([1, 1] if condicao == "TSST" else [])
        panas[20] = estressado[condicao]
        linhas.append("# PANAS;" + ";".join(map(str, panas)) + ";")
    for condicao in condicoes:
        # TSST: totalmente ansioso (itens positivos = 1, negativos = 4); resto: totalmente calmo.
        stai = [1, 4, 4, 1, 4, 1] if condicao == "TSST" else [4, 1, 1, 4, 1, 4]
        linhas.append("# STAI;" + ";".join(map(str, stai)) + ";;")
    for condicao in condicoes:
        linhas.append("# DIM;" + ("3;8" if condicao == "TSST" else "7;2") + ";;")
    linhas.append("# SSSQ;5;5;4;4;3;5;;")
    (pasta / f"{participante}_quest.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def test_ler_relatos_wesad_segue_a_ordem_de_cada_participante(tmp_path):
    escrever_quest(tmp_path, "S2", ["Base", "TSST", "Medi 1", "Fun", "Medi 2"],
                   {"Base": 1, "TSST": 4, "Medi 1": 1, "Fun": 2, "Medi 2": 1})
    escrever_quest(tmp_path, "S3", ["Base", "Fun", "Medi 1", "TSST", "Medi 2"],
                   {"Base": 2, "TSST": 5, "Medi 1": 1, "Fun": 1, "Medi 2": 1})
    relatos = ler_relatos_wesad(tmp_path).set_index(["participante", "fase"])

    assert relatos.loc[("S2", "TSST"), "relato"] == 4
    assert relatos.loc[("S3", "TSST"), "relato"] == 5      # TSST é a 4ª condição do S3
    assert relatos.loc[("S3", "Fun"), "relato"] == 1
    assert relatos.loc[("S2", "TSST"), "ansiedade_stai"] == 80.0   # máximo da escala
    assert relatos.loc[("S2", "Base"), "ansiedade_stai"] == 20.0   # mínimo
    assert relatos.loc[("S3", "TSST"), "ativacao_sam"] == 8
    assert len(relatos) == 10


def test_ler_relatos_physionet_junta_as_tarefas_curtas(tmp_path):
    (tmp_path / "Stress_Level_v1.csv").write_text(
        ",Baseline,Stroop,First Rest,TMCT,Second Rest,Real Opinion,Opposite Opinion,Subtract\n"
        "S01,3.00,4.50,4.00,4.00,5.50,6.00,7.00,8.00\n", encoding="utf-8")
    (tmp_path / "Stress_Level_v2.csv").write_text(
        ",Baseline,TMCT,First Rest,Real Opinion,Opposite Opinion,Second Rest,Subtract\n"
        "f01,2.00,5.00,2.00,3.00,4.00,2.00,3.00\n", encoding="utf-8")
    relatos = ler_relatos_physionet(tmp_path).set_index(["participante", "fase"]).relato

    assert relatos[("pn-S01", "Opinioes e contas")] == pytest.approx(7.0)   # média de 6, 7 e 8
    assert relatos[("pn-f01", "Opinioes")] == pytest.approx(3.5)            # média de 3 e 4
    assert relatos[("pn-f01", "TMCT")] == 5.0
    assert ("pn-f01", "Subtract") not in relatos.index   # depois do 2º descanso: sem janelas


def test_correlacao_dentro_da_pessoa_ignora_o_nivel_de_cada_um():
    # Dentro de cada pessoa, y sobe com x; entre pessoas, os níveis são opostos.
    tabela = pd.DataFrame({
        "participante": ["a"] * 3 + ["b"] * 3,
        "x": [1, 2, 3, 11, 12, 13],
        "y": [10, 11, 12, 0, 1, 2],
    })
    assert correlacao_dentro_da_pessoa(tabela, "x", "y")["rho"] == pytest.approx(1.0)


@pytest.fixture
def tabela():
    """Duas pessoas no PhysioNet: o modelo alerta no Stroop, mas só uma diz ter sentido."""
    janelas = []
    for participante, prob_stroop in (("pn-S01", 0.9), ("pn-S03", 0.8)):
        for fase, estado, prob, fc in (("First Rest", "repouso", 0.1, -2.0), ("Stroop", "estresse", prob_stroop, 8.0),
                                       ("TMCT", "estresse", 0.2, 3.0)):
            for _ in range(4):
                janelas.append({"participante": participante, "base": "physionet", "fase": fase,
                                "estado": estado, "fc_media": fc, "eda_media": 0.1,
                                "prob_modelo_wesad": prob, "prob_modelo_da_base": prob})
    relatos = pd.DataFrame([
        {"participante": p, "base": "physionet", "fase": f, "relato": r}
        for p, notas in (("pn-S01", {"Baseline": 3, "First Rest": 3, "Stroop": 6, "TMCT": 7}),
                         ("pn-S03", {"Baseline": 4, "First Rest": 3, "Stroop": 4, "TMCT": 8}))
        for f, r in notas.items()
    ])
    return por_fase(pd.DataFrame(janelas), relatos)


def test_por_fase_calcula_mudanca_do_relato_e_alerta(tabela):
    linha = tabela.set_index(["participante", "fase"]).loc[("pn-S01", "Stroop")]
    assert linha.mudanca_relato == 3          # 6 contra 3 na linha de base
    assert linha.alerta_pct == 100.0
    assert linha.protocolo_estresse == 1
    assert len(tabela) == 6                    # a linha de base não tem janelas (é a calibração)


def test_analisar_separa_fases_pelo_relato(tabela):
    resultado = analisar(tabela)["physionet"]
    assert resultado["participantes"] == 2
    # Fases de estresse: S01 sentiu nas duas, S03 só no TMCT -> 3 de 4.
    assert resultado["fases_de_estresse_com_relato_acima_da_linha_de_base_pct"] == 75.0
    assert resultado["outras_fases_com_relato_acima_da_linha_de_base_pct"] == 0.0
    estresse = resultado["por_relato"]["fases_de_estresse"]
    assert estresse["relato_acima_da_linha_de_base"]["fases"] == 3
    assert estresse["relato_igual_ou_abaixo"]["janelas_com_alerta_pct"] == 100.0   # o Stroop do S03
    assert resultado["por_fase"]["TMCT"]["relato_acima_da_linha_de_base_pct"] == 100.0
    assert "protocolo_estresse x relato" in resultado["correlacao_dentro_da_pessoa"]
