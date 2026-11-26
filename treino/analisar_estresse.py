"""O que o modelo chama de estresse? Compara três definições de estresse.

1. Protocolo: a fase do experimento (o rótulo usado no treino).
2. Relato: o que a pessoa disse sentir depois de cada fase.
   - WESAD: item "Stressed" do PANAS (1 a 5), STAI curto (ansiedade, 20 a 80) e
     SAM (ativação e valência, 1 a 9), respondidos depois de cada condição.
   - PhysioNet: nível de estresse de 1 a 10 em cada fase (Stress_Level_v1/v2.csv).
3. Corpo e modelo: a reação fisiológica (frequência cardíaca e condutância, em
   relação à calibração da pessoa) e a probabilidade de estresse dada pelo modelo.

Perguntas:
- O protocolo coincide com o relato? (a pessoa se sentiu mais estressada nas fases
  de estresse do que na linha de base?)
- O corpo e o modelo acompanham mais o protocolo ou o relato?
- Quando a pessoa diz que não se sentiu estressada numa fase "de estresse", o
  modelo também não alerta?

O Stress-Predict fica de fora: só tem questionários antes e depois da sessão inteira.

Uso (na raiz, depois de preparar o WESAD e o PhysioNet):
    python -m treino.analisar_estresse --physionet-relatos <pasta com Stress_Level_v1.csv>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from treino.avaliar_bases import carregar
from treino.treinar import AZUL, FEATURES, FUNDO, GRADE, LARANJA, TEXTO, TEXTO_SECUNDARIO, _estilo, novo_modelo

# Itens do STAI curto, na ordem do SX_quest.csv (readme do WESAD); True = item positivo,
# pontuado ao contrário (5 - resposta).
ITENS_STAI = [("à vontade", True), ("nervoso", False), ("agitado", False),
              ("relaxado", True), ("preocupado", False), ("bem", True)]
POSICAO_PANAS_STRESSED = 20  # 21º item do PANAS: "Stressed"
CONDICOES_WESAD = ("Base", "TSST", "Medi 1", "Fun", "Medi 2")

# Fases do PhysioNet no relato -> fases da tabela de features (treino.preparar_physionet).
FASES_RELATO_PHYSIONET = {
    "v1": {"Baseline": ["Baseline"], "Stroop": ["Stroop"], "First Rest": ["First Rest"], "TMCT": ["TMCT"],
           "Second Rest": ["Second Rest"], "Opinioes e contas": ["Real Opinion", "Opposite Opinion", "Subtract"]},
    "v2": {"Baseline": ["Baseline"], "TMCT": ["TMCT"], "First Rest": ["First Rest"],
           "Opinioes": ["Real Opinion", "Opposite Opinion"], "Second Rest": ["Second Rest"]},
}
LINHA_DE_BASE = {"wesad": "Base", "physionet": "Baseline"}


# --- relatos ---------------------------------------------------------------------------


def ler_relatos_wesad(pasta: Path) -> pd.DataFrame:
    """Uma linha por participante e condição, com os questionários do SX_quest.csv."""
    linhas = []
    for arquivo in sorted(pasta.glob("S*/S*_quest.csv")):
        blocos: dict[str, list[list[float]]] = {}
        ordem = []
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            campos = [c.strip() for c in linha.split(";")]
            if not campos[0].startswith("#"):
                continue
            nome, valores = campos[0].lstrip("# ").strip(), [c for c in campos[1:] if c]
            if nome == "ORDER":
                ordem = [c for c in valores if c in CONDICOES_WESAD]
            elif nome in ("PANAS", "STAI", "DIM"):
                blocos.setdefault(nome, []).append([float(v) for v in valores])
        if sorted(ordem) != sorted(CONDICOES_WESAD):
            raise ValueError(f"{arquivo}: ordem das condições inesperada: {ordem}")
        for posicao, condicao in enumerate(ordem):
            stai = blocos["STAI"][posicao]
            ansiedade = sum(5 - v if positivo else v for v, (_, positivo) in zip(stai, ITENS_STAI, strict=True))
            valencia, ativacao = blocos["DIM"][posicao][:2]
            linhas.append({
                "participante": arquivo.parent.name,
                "base": "wesad",
                "fase": condicao,
                "relato": blocos["PANAS"][posicao][POSICAO_PANAS_STRESSED],  # "Stressed", 1 a 5
                "ansiedade_stai": round(ansiedade * 20 / 6, 1),
                "ativacao_sam": ativacao,
                "valencia_sam": valencia,
            })
    return pd.DataFrame(linhas)


def ler_relatos_physionet(pasta: Path) -> pd.DataFrame:
    """Uma linha por participante e fase, com o estresse relatado (1 a 10)."""
    linhas = []
    for versao, fases in FASES_RELATO_PHYSIONET.items():
        tabela = pd.read_csv(pasta / f"Stress_Level_{versao}.csv", index_col=0)
        for participante, respostas in tabela.iterrows():
            for fase, colunas in fases.items():
                valores = respostas[colunas].astype(float)
                if valores.isna().any():
                    continue
                linhas.append({"participante": f"pn-{participante.strip()}", "base": "physionet",
                               "fase": fase, "relato": float(valores.mean())})
    return pd.DataFrame(linhas)


# --- corpo e modelo ----------------------------------------------------------------------


def probabilidades(dados: pd.DataFrame, n_arvores: int) -> pd.DataFrame:
    """Probabilidade de estresse de cada janela, sem o modelo ter visto a pessoa.

    - WESAD: leave-one-subject-out no WESAD (o modelo da API, avaliado com honestidade);
    - PhysioNet: o modelo treinado em todo o WESAD (o que a API faria com essas
      pessoas) e, para comparação, leave-one-subject-out no próprio PhysioNet.
    """
    wesad = dados[dados.base == "wesad"]
    pn = dados[(dados.base == "physionet") & (dados.sessao == "STRESS")]
    logo = LeaveOneGroupOut()

    def loso(conjunto: pd.DataFrame) -> pd.Series:
        prob = cross_val_predict(novo_modelo(n_arvores), conjunto[FEATURES], conjunto.estado,
                                 groups=conjunto.participante, cv=logo, method="predict_proba")
        return pd.Series(prob[:, sorted(conjunto.estado.unique()).index("estresse")], index=conjunto.index)

    modelo_wesad = novo_modelo(n_arvores).fit(wesad[FEATURES].to_numpy(), wesad.estado.to_numpy())
    classe = list(modelo_wesad.classes_).index("estresse")
    prob_wesad = loso(wesad)
    return pd.concat([wesad, pn]).assign(
        prob_modelo_wesad=pd.concat([
            prob_wesad,
            pd.Series(modelo_wesad.predict_proba(pn[FEATURES].to_numpy())[:, classe], index=pn.index),
        ]),
        prob_modelo_da_base=pd.concat([prob_wesad, loso(pn)]),
    )


def por_fase(janelas: pd.DataFrame, relatos: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por participante e fase: protocolo, relato, reação do corpo e modelo."""
    agregado = janelas.groupby(["participante", "base", "fase"]).agg(
        estado=("estado", "first"),
        janelas=("estado", "size"),
        delta_fc=("fc_media", "mean"),
        delta_eda=("eda_media", "mean"),
        prob_modelo_wesad=("prob_modelo_wesad", "mean"),
        prob_modelo_da_base=("prob_modelo_da_base", "mean"),
        alerta_pct=("prob_modelo_wesad", lambda p: 100 * float(np.mean(p >= 0.5))),
    ).reset_index()

    base_relato = relatos[relatos.fase == relatos.base.map(LINHA_DE_BASE)].set_index("participante").relato
    relatos = relatos.assign(mudanca_relato=relatos.relato - relatos.participante.map(base_relato))
    tabela = agregado.merge(relatos, on=["participante", "base", "fase"], how="inner")
    return tabela.assign(protocolo_estresse=(tabela.estado == "estresse").astype(int))


# --- análises ----------------------------------------------------------------------------


def correlacao_dentro_da_pessoa(tabela: pd.DataFrame, x: str, y: str) -> dict:
    """Spearman entre x e y depois de tirar a média de cada pessoa (compara a pessoa com ela mesma)."""
    centrado = tabela[[x, y]] - tabela.groupby("participante")[[x, y]].transform("mean")
    rho, p = spearmanr(centrado[x], centrado[y])
    return {"rho": round(float(rho), 2), "p": float(f"{p:.2g}"), "n": int(len(tabela))}


def analisar(tabela: pd.DataFrame) -> dict:
    resultado = {}
    for base, grupo in tabela.groupby("base"):
        fora_da_base = grupo[grupo.fase != LINHA_DE_BASE[base]]
        estresse = fora_da_base[fora_da_base.protocolo_estresse == 1]
        repouso = fora_da_base[fora_da_base.protocolo_estresse == 0]

        modelos = ["prob_modelo_wesad"] + (["prob_modelo_da_base"] if base == "physionet" else [])
        medidas = ["protocolo_estresse", "relato", "delta_fc", "delta_eda"] + modelos
        resultado[base] = {
            "participantes": int(grupo.participante.nunique()),
            "por_fase": {
                fase: {
                    "pessoas": int(len(g)),
                    "estado_no_protocolo": g.estado.iloc[0],
                    "relato_medio": round(float(g.relato.mean()), 2),
                    "relato_acima_da_linha_de_base_pct": round(100 * float((g.mudanca_relato > 0).mean()), 1),
                    "delta_fc_medio": round(float(g.delta_fc.mean()), 1),
                    "delta_eda_medio": round(float(g.delta_eda.mean()), 2),
                    "prob_estresse_media": round(float(g.prob_modelo_wesad.mean()), 2),
                    "janelas_com_alerta_pct": round(float(g.alerta_pct.mean()), 1),
                }
                for fase, g in grupo.groupby("fase", sort=False)
            },
            # Protocolo x relato: nas fases "de estresse", quantas vezes a pessoa relatou mais
            # estresse que na linha de base? E nos descansos/diversão?
            "fases_de_estresse_com_relato_acima_da_linha_de_base_pct": round(
                100 * float((estresse.mudanca_relato > 0).mean()), 1),
            "outras_fases_com_relato_acima_da_linha_de_base_pct": round(
                100 * float((repouso.mudanca_relato > 0).mean()), 1) if len(repouso) else None,
            # O corpo e o modelo acompanham o protocolo ou o relato? (pessoa contra ela mesma)
            "correlacao_dentro_da_pessoa": {
                f"{a} x {b}": correlacao_dentro_da_pessoa(grupo, a, b)
                for i, a in enumerate(medidas) for b in medidas[i + 1:]
            },
            # O modelo alerta mais quando a pessoa sentiu o estresse? Separado por tipo de fase.
            "por_relato": {
                tipo: {
                    rotulo: {
                        "fases": int(len(g)),
                        "janelas_com_alerta_pct": round(float(g.alerta_pct.mean()), 1),
                        "delta_fc_medio": round(float(g.delta_fc.mean()), 1),
                        "delta_eda_medio": round(float(g.delta_eda.mean()), 2),
                    }
                    for rotulo, g in (("relato_acima_da_linha_de_base", fases[fases.mudanca_relato > 0]),
                                      ("relato_igual_ou_abaixo", fases[fases.mudanca_relato <= 0]))
                    if len(g)
                }
                for tipo, fases in (("fases_de_estresse", estresse), ("outras_fases", repouso))
            },
        }
        if base == "wesad":
            colunas = ["relato", "ansiedade_stai", "ativacao_sam", "valencia_sam"]
            resultado[base]["questionarios_por_fase"] = (
                grupo.groupby("fase")[colunas].mean().round(2).to_dict(orient="index"))
    return resultado


# --- gráfico -----------------------------------------------------------------------------

# (base, fase na tabela, rótulo no gráfico), na ordem em que aparecem.
FASES_GRAFICO = [
    ("wesad", "TSST", "Estresse\n(TSST)"),
    ("wesad", "Fun", "Diversão"),
    ("physionet", "Stroop", "Stroop"),
    ("physionet", "TMCT", "Contas\n(TMCT)"),
    ("physionet", "Opinioes e contas", "Opiniões\ne contas"),
    ("physionet", "Opinioes", "Opiniões"),
    ("physionet", "First Rest", "1º\ndescanso"),
    ("physionet", "Second Rest", "2º\ndescanso"),
]


def gerar_grafico(resultado: dict, arquivo: Path) -> Path:
    """Por fase: quantos relataram mais estresse que no repouso x quanto o modelo alertou."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "text.color": TEXTO, "axes.labelcolor": TEXTO_SECUNDARIO})
    fases = [(b, f, r) for b, f, r in FASES_GRAFICO if f in resultado.get(b, {}).get("por_fase", {})]
    sentiu = [resultado[b]["por_fase"][f]["relato_acima_da_linha_de_base_pct"] for b, f, _ in fases]
    alerta = [resultado[b]["por_fase"][f]["janelas_com_alerta_pct"] for b, f, _ in fases]
    # Um espaço entre as duas bases.
    x = np.array([i + (0.6 if b == "physionet" else 0) for i, (b, _, _) in enumerate(fases)])

    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=FUNDO)
    _estilo(ax)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    largura = 0.36
    ax.bar(x - largura / 2 - 0.01, sentiu, largura, color=AZUL, edgecolor=FUNDO, linewidth=1,
           label="Pessoas que relataram mais estresse que na linha de base")
    ax.bar(x + largura / 2 + 0.01, alerta, largura, color=LARANJA, edgecolor=FUNDO, linewidth=1,
           label="Janelas que o modelo da API classificou como estresse")
    ax.set_xticks(x, [r for _, _, r in fases], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    for indice, (b, f, _) in enumerate(fases):
        if f in ("TMCT", "Fun"):  # o contraste principal e uma barra vazia (0%)
            for deslocamento, valor in ((-largura / 2, sentiu[indice]), (largura / 2, alerta[indice])):
                ax.annotate(f"{valor:.0f}%", (x[indice] + deslocamento, valor), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=9, color=TEXTO)
    for base, nome in (("wesad", "WESAD (estresse intenso)"), ("physionet", "PhysioNet (estresse leve)")):
        posicoes = [x[i] for i, (b, _, _) in enumerate(fases) if b == base]
        if posicoes:
            ax.text(np.mean(posicoes), 101, nome, ha="center", fontsize=9, color=TEXTO_SECUNDARIO)
    ax.set_title("Por fase, o alerta acompanha o relato, com uma exceção: as contas (TMCT)",
                 loc="left", fontsize=11, color=TEXTO)
    ax.legend(loc="upper right", bbox_to_anchor=(1, -0.2), frameon=False, fontsize=9,
              labelcolor=TEXTO_SECUNDARIO, ncol=1)
    fig.tight_layout()
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(arquivo, dpi=160, facecolor=FUNDO)
    plt.close(fig)
    return arquivo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wesad", type=Path, default=Path("data/features_wesad.csv"))
    parser.add_argument("--wesad-relatos", type=Path, default=Path("WESAD"), help="pasta do WESAD (SX_quest.csv)")
    parser.add_argument("--physionet", type=Path, default=Path("data/features_physionet.csv"))
    parser.add_argument("--physionet-relatos", type=Path, required=True,
                        help="pasta do PhysioNet com Stress_Level_v1.csv e Stress_Level_v2.csv")
    parser.add_argument("--saida", type=Path, default=Path("resultados/o_que_e_estresse.json"))
    parser.add_argument("--tabela", type=Path, default=Path("data/estresse_por_fase.csv"))
    parser.add_argument("--grafico", type=Path, default=Path("resultados/relato_x_alerta.png"))
    parser.add_argument("--arvores", type=int, default=300)
    args = parser.parse_args()

    relatos = pd.concat([ler_relatos_wesad(args.wesad_relatos), ler_relatos_physionet(args.physionet_relatos)],
                        ignore_index=True)
    janelas = probabilidades(carregar(args.wesad, args.physionet), args.arvores)
    tabela = por_fase(janelas, relatos)
    args.tabela.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(args.tabela, index=False)

    resultado = analisar(tabela)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    gerar_grafico(resultado, args.grafico)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(f"Tabela: {args.tabela} | gráfico: {args.grafico}")


if __name__ == "__main__":
    main()
