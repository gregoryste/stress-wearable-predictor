"""Compara modelos treinados em bases diferentes (WESAD, PhysioNet e Stress-Predict).

Perguntas sobre WESAD e PhysioNet:
1. Teste cruzado: um modelo treinado numa base funciona na outra?
2. Exercício: quanto o modelo atual (só WESAD) confunde exercício com estresse?
3. Modelo combinado (WESAD + PhysioNet, com a classe "exercicio"), avaliado com
   leave-one-subject-out nas duas bases: o falso alarme no exercício cai sem perder
   a detecção de estresse?

Com a terceira base (Stress-Predict), sem a classe "exercicio":
4. Dentro de cada base (leave-one-subject-out), qual o desempenho de referência?
5. Treinando em uma ou duas bases e testando numa base que o modelo nunca viu,
   juntar bases melhora a generalização?
6. Com as três bases juntas (leave-one-subject-out), cada base melhora ou piora em
   relação ao modelo treinado só nela?

Todas as variantes usam o padrão pessoal (diferença para a calibração de cada
participante); as janelas de calibração ficam fora da avaliação.

Uso (na raiz, depois de preparar as bases; o Stress-Predict é opcional):
    python -m treino.avaliar_bases
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from treino.treinar import FEATURES, estatisticas_calibracao, marcar_calibracao, normalizar, novo_modelo


BASES = ("wesad", "physionet", "stress_predict")
# Nome das condições do WESAD no arquivo de questionários (SX_quest.csv).
FASES_WESAD = {"repouso": "Base", "estresse": "TSST", "diversao": "Fun"}


def carregar(tabela_wesad: Path, tabela_physionet: Path, tabela_stress_predict: Path | None = None) -> pd.DataFrame:
    wesad = pd.read_csv(tabela_wesad)
    calibracao, sobreposta = marcar_calibracao(wesad)
    wesad = wesad.assign(base="wesad", sessao="WESAD", fase=wesad.estado.map(FASES_WESAD),
                         calibracao=calibracao, fora=sobreposta)
    # As outras bases já trazem a calibração marcada pelo preparador.
    outras = [pd.read_csv(t).assign(fora=False) for t in (tabela_physionet, tabela_stress_predict) if t is not None]
    colunas = ["participante", "base", "sessao", "fase", "estado", "calibracao", "fora"] + FEATURES
    dados = pd.concat([wesad[colunas]] + [o[colunas] for o in outras], ignore_index=True)

    # Padrão pessoal: só participantes com calibração (no PhysioNet, quem tem a sessão de estresse).
    estatisticas = estatisticas_calibracao(dados, dados.calibracao.astype(bool))
    dados = dados[dados.participante.isin(estatisticas)].reset_index(drop=True)
    dados[FEATURES] = normalizar(dados, estatisticas)
    return dados[~dados.calibracao.astype(bool) & ~dados.fora].reset_index(drop=True)


def binario(verdadeiro, previsto, prob_estresse=None) -> dict:
    """Estresse contra o resto."""
    v, p = np.asarray(verdadeiro) == "estresse", np.asarray(previsto) == "estresse"
    r = {
        "janelas": int(len(v)),
        "f1_estresse": round(100 * f1_score(v, p, zero_division=0), 1),
        "acuracia_balanceada": round(100 * balanced_accuracy_score(v, p), 1),
    }
    if prob_estresse is not None and v.any() and not v.all():
        r["auc"] = round(float(roc_auc_score(v, prob_estresse)), 3)
    return r


def prob_classe(modelo, X, classe: str) -> np.ndarray:
    return modelo.predict_proba(X)[:, list(modelo.classes_).index(classe)]


def avaliar(dados: pd.DataFrame, n_arvores: int) -> dict:
    wesad = dados[dados.base == "wesad"]
    pn = dados[dados.base == "physionet"]
    pn_estresse = pn[pn.sessao == "STRESS"]          # repouso + estresse
    pn_exercicio = pn[pn.estado == "exercicio"]
    resultado = {
        "janelas": {
            "wesad": wesad.estado.value_counts().to_dict(),
            "physionet": pn.estado.value_counts().to_dict(),
            "participantes": {"wesad": int(wesad.participante.nunique()),
                              "physionet": int(pn.participante.nunique())},
        }
    }

    # 1. Teste cruzado (sem exercício): WESAD -> PhysioNet e PhysioNet -> WESAD.
    so_wesad = novo_modelo(n_arvores).fit(wesad[FEATURES].to_numpy(), wesad.estado.to_numpy())
    so_pn = novo_modelo(n_arvores).fit(pn_estresse[FEATURES].to_numpy(), pn_estresse.estado.to_numpy())
    resultado["cruzado"] = {
        "treino_wesad_teste_physionet": binario(
            pn_estresse.estado, so_wesad.predict(pn_estresse[FEATURES].to_numpy()),
            prob_classe(so_wesad, pn_estresse[FEATURES].to_numpy(), "estresse"),
        ),
        "treino_physionet_teste_wesad": binario(
            wesad.estado, so_pn.predict(wesad[FEATURES].to_numpy()),
            prob_classe(so_pn, wesad[FEATURES].to_numpy(), "estresse"),
        ),
    }

    # 2. Modelo atual (só WESAD) diante do exercício.
    previsto_exercicio = so_wesad.predict(pn_exercicio[FEATURES].to_numpy())
    resultado["exercicio_modelo_so_wesad"] = {
        "janelas": int(len(pn_exercicio)),
        "chamado_de_estresse_pct": round(100 * float(np.mean(previsto_exercicio == "estresse")), 1),
    }

    # 3. Leave-one-subject-out: só WESAD x combinado (WESAD + PhysioNet com exercício).
    logo = LeaveOneGroupOut()
    prev_so_wesad = cross_val_predict(novo_modelo(n_arvores), wesad[FEATURES], wesad.estado,
                                      groups=wesad.participante, cv=logo)
    probabilidades = pd.DataFrame(
        cross_val_predict(novo_modelo(n_arvores), dados[FEATURES], dados.estado,
                          groups=dados.participante, cv=logo, method="predict_proba"),
        columns=sorted(dados.estado.unique()),
        index=dados.index,
    )
    prev_comb = probabilidades.idxmax(axis=1)
    resultado["loso"] = {
        "so_wesad_no_wesad": binario(wesad.estado, prev_so_wesad),
        "combinado_no_wesad": binario(wesad.estado, prev_comb.loc[wesad.index]),
        "combinado_no_physionet_estresse": binario(pn_estresse.estado, prev_comb.loc[pn_estresse.index]),
        "combinado_exercicio": {
            "janelas": int(len(pn_exercicio)),
            "chamado_de_estresse_pct": round(
                100 * float((prev_comb.loc[pn_exercicio.index] == "estresse").mean()), 1),
            "reconhecido_como_exercicio_pct": round(
                100 * float((prev_comb.loc[pn_exercicio.index] == "exercicio").mean()), 1),
        },
        "combinado_wesad_estresse_chamado_de_exercicio_pct": round(
            100 * float((prev_comb.loc[wesad[wesad.estado == "estresse"].index] == "exercicio").mean()), 1),
    }

    # 4. Meio-termo: o estresse é decidido sem a classe exercício, que só suprime o
    # alerta quando sua probabilidade passa de um limiar.
    sem_exercicio = probabilidades.drop(columns="exercicio").idxmax(axis=1)
    resultado["supressao_por_exercicio"] = []
    for limiar in (0.5, 0.7, 0.8, 0.9, None):
        suprime = probabilidades.exercicio >= limiar if limiar is not None else False
        estresse = (sem_exercicio == "estresse") & ~suprime
        previsto = np.where(estresse, "estresse", "outro")
        resultado["supressao_por_exercicio"].append({
            "limiar_exercicio": limiar if limiar is not None else "nunca suprime",
            "f1_estresse_wesad": binario(wesad.estado, previsto[dados.base == "wesad"])["f1_estresse"],
            "f1_estresse_physionet": binario(
                pn_estresse.estado, previsto[(dados.base == "physionet") & (dados.sessao == "STRESS")]
            )["f1_estresse"],
            "exercicio_chamado_de_estresse_pct": round(100 * float(estresse[pn_exercicio.index].mean()), 1),
        })
    return resultado


def avaliar_entre_bases(dados: pd.DataFrame, n_arvores: int) -> dict:
    """Perguntas 4 a 6: generalização entre bases, sem a classe exercício.

    Nos testes no PhysioNet, também mede quanto do exercício (nunca visto no treino)
    o modelo chama de estresse.
    """
    exercicio = dados[dados.estado == "exercicio"]
    dados = dados[dados.estado != "exercicio"].reset_index(drop=True)
    bases = [b for b in BASES if b in set(dados.base)]
    por_base = {b: dados[dados.base == b] for b in bases}
    logo = LeaveOneGroupOut()

    def loso(conjunto: pd.DataFrame) -> pd.DataFrame:
        probabilidades = cross_val_predict(novo_modelo(n_arvores), conjunto[FEATURES], conjunto.estado,
                                           groups=conjunto.participante, cv=logo, method="predict_proba")
        return pd.DataFrame(probabilidades, columns=sorted(conjunto.estado.unique()), index=conjunto.index)

    def medir(verdadeiro, probabilidades: pd.DataFrame) -> dict:
        return binario(verdadeiro, probabilidades.idxmax(axis=1), probabilidades.estresse)

    resultado = {
        "participantes": {b: int(g.participante.nunique()) for b, g in por_base.items()},
        "janelas": {b: g.estado.value_counts().to_dict() for b, g in por_base.items()},
        "dentro_da_base": {b: medir(g.estado, loso(g)) for b, g in por_base.items()},
        "base_nunca_vista": [],
    }

    for teste in bases:
        outras = [b for b in bases if b != teste]
        X_teste = por_base[teste][FEATURES].to_numpy()
        for tamanho in range(1, len(outras) + 1):
            for treino in itertools.combinations(outras, tamanho):
                conjunto = dados[dados.base.isin(treino)]
                modelo = novo_modelo(n_arvores).fit(conjunto[FEATURES].to_numpy(), conjunto.estado.to_numpy())
                medida = {
                    "treino": " + ".join(treino),
                    "teste": teste,
                    **binario(por_base[teste].estado, modelo.predict(X_teste), prob_classe(modelo, X_teste, "estresse")),
                }
                if teste == "physionet" and len(exercicio):
                    previsto = modelo.predict(exercicio[FEATURES].to_numpy())
                    medida["exercicio_chamado_de_estresse_pct"] = round(100 * float(np.mean(previsto == "estresse")), 1)
                resultado["base_nunca_vista"].append(medida)

    todas = loso(dados)
    resultado["todas_juntas_loso"] = {b: medir(g.estado, todas.loc[g.index]) for b, g in por_base.items()}
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wesad", type=Path, default=Path("data/features_wesad.csv"))
    parser.add_argument("--physionet", type=Path, default=Path("data/features_physionet.csv"))
    parser.add_argument("--stress-predict", type=Path, default=Path("data/features_stress_predict.csv"),
                        help="tabela do Stress-Predict; se não existir, as perguntas 4 a 6 ficam de fora")
    parser.add_argument("--saida", type=Path, default=Path("resultados/comparacao_bases.json"))
    parser.add_argument("--arvores", type=int, default=300)
    args = parser.parse_args()
    for tabela, comando in ((args.wesad, "treino.preparar_wesad"), (args.physionet, "treino.preparar_physionet")):
        if not tabela.exists():
            parser.error(f"{tabela} não existe. Rode antes: python -m {comando}")

    stress_predict = args.stress_predict if args.stress_predict.exists() else None
    dados = carregar(args.wesad, args.physionet, stress_predict)
    # Perguntas 1 a 3: só WESAD e PhysioNet, como na primeira comparação.
    resultado = avaliar(dados[dados.base != "stress_predict"].reset_index(drop=True), args.arvores)
    if stress_predict is not None:
        resultado["entre_bases"] = avaliar_entre_bases(dados, args.arvores)
    else:
        print(f"{args.stress_predict} não existe: perguntas 4 a 6 puladas (python -m treino.preparar_stress_predict)")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
