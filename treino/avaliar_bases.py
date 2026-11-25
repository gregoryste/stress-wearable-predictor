"""Compara o modelo treinado só no WESAD com um modelo que usa as duas bases.

Perguntas:
1. Teste cruzado: um modelo treinado numa base funciona na outra?
2. Exercício: quanto o modelo atual (só WESAD) confunde exercício com estresse?
3. Modelo combinado (WESAD + PhysioNet, com a classe "exercicio"), avaliado com
   leave-one-subject-out nas duas bases: o falso alarme no exercício cai sem perder
   a detecção de estresse?

Todas as variantes usam o padrão pessoal (diferença para a calibração de cada
participante); as janelas de calibração ficam fora da avaliação.

Uso (na raiz, depois de preparar as duas bases):
    python -m treino.avaliar_bases
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from treino.treinar import FEATURES, estatisticas_calibracao, marcar_calibracao, normalizar, novo_modelo


def carregar(tabela_wesad: Path, tabela_physionet: Path) -> pd.DataFrame:
    wesad = pd.read_csv(tabela_wesad)
    calibracao, sobreposta = marcar_calibracao(wesad)
    wesad = wesad.assign(base="wesad", sessao="WESAD", calibracao=calibracao, fora=sobreposta)
    physionet = pd.read_csv(tabela_physionet).assign(fora=False)
    colunas = ["participante", "base", "sessao", "estado", "calibracao", "fora"] + FEATURES
    dados = pd.concat([wesad[colunas], physionet[colunas]], ignore_index=True)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wesad", type=Path, default=Path("data/features_wesad.csv"))
    parser.add_argument("--physionet", type=Path, default=Path("data/features_physionet.csv"))
    parser.add_argument("--saida", type=Path, default=Path("resultados/comparacao_bases.json"))
    parser.add_argument("--arvores", type=int, default=300)
    args = parser.parse_args()
    for tabela, comando in ((args.wesad, "treino.preparar_wesad"), (args.physionet, "treino.preparar_physionet")):
        if not tabela.exists():
            parser.error(f"{tabela} não existe. Rode antes: python -m {comando}")

    resultado = avaliar(carregar(args.wesad, args.physionet), args.arvores)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
