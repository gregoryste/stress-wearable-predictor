"""Compara modelos de classificação no mesmo protocolo do treino.

Mesmas janelas, mesma calibração pessoal (diferença para os 5 minutos de repouso de
cada participante) e a mesma validação leave-one-subject-out de treino/treinar.py.
Muda só o modelo:

- Random Forest (o modelo atual) e variações de tamanho mínimo de folha;
- Extra Trees (árvores mais aleatórias);
- Gradient Boosting por histograma (árvores construídas em sequência, cada uma
  corrigindo os erros das anteriores);
- Regressão logística (modelo linear, como referência simples);
- k vizinhos mais próximos.

Atenção: escolher o melhor modelo olhando o mesmo resultado de validação deixa a
escolha levemente otimista. Diferenças de 1 ou 2 pontos não justificam trocar.

Uso (na raiz, depois de preparar o WESAD):
    python -m treino.comparar_modelos
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from treino.treinar import (
    FEATURES,
    SEMENTE,
    estatisticas_calibracao,
    marcar_calibracao,
    metricas,
    normalizar,
)


def candidatos() -> dict:
    floresta = dict(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEMENTE)
    return {
        "Random Forest (atual, folha mínima 3)": RandomForestClassifier(min_samples_leaf=3, **floresta),
        "Random Forest, folha mínima 1": RandomForestClassifier(min_samples_leaf=1, **floresta),
        "Random Forest, folha mínima 10": RandomForestClassifier(min_samples_leaf=10, **floresta),
        "Extra Trees": ExtraTreesClassifier(min_samples_leaf=3, **floresta),
        "Gradient Boosting (histograma)": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEMENTE
        ),
        "Regressão logística": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "k vizinhos (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }


def comparar(tabela: Path) -> dict:
    df = pd.read_csv(tabela)
    calibracao, sobreposta = marcar_calibracao(df)
    avaliacao = df[~calibracao & ~sobreposta]
    X = normalizar(avaliacao, estatisticas_calibracao(df, calibracao)).to_numpy()
    y, grupos = avaliacao.estado.to_numpy(), avaliacao.participante.to_numpy()

    resultado = {}
    for nome, modelo in candidatos().items():
        inicio = time.perf_counter()
        previsto = cross_val_predict(modelo, X, y, groups=grupos, cv=LeaveOneGroupOut())
        m = metricas(y, previsto, grupos)
        f1_participantes = np.array(list(m["f1_por_participante"].values()))
        resultado[nome] = {
            "f1_macro": m["f1_macro"],
            "acuracia": m["acuracia"],
            "f1_estresse": m["f1_por_classe"]["estresse"],
            "f1_diversao": m["f1_por_classe"]["diversao"],
            "f1_pior_participante": round(float(f1_participantes.min()), 2),
            "f1_desvio_entre_participantes": round(float(f1_participantes.std()), 2),
            "tempo_s": round(time.perf_counter() - inicio, 1),
        }
        print(f"{nome:38s} F1 {m['f1_macro']:5.1f} | acurácia {m['acuracia']:5.1f} | "
              f"estresse {m['f1_por_classe']['estresse']:5.1f} | pior participante "
              f"{f1_participantes.min():5.1f} | {resultado[nome]['tempo_s']} s")
    return {
        "protocolo": "padrão pessoal (5 min de calibração), leave-one-subject-out, "
                     f"{len(avaliacao)} janelas, {len(set(grupos))} participantes",
        "modelos": resultado,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tabela", type=Path, default=Path("data/features_wesad.csv"))
    parser.add_argument("--saida", type=Path, default=Path("resultados/comparacao_modelos.json"))
    args = parser.parse_args()
    if not args.tabela.exists():
        parser.error(f"{args.tabela} não existe. Rode antes: python -m treino.preparar_wesad")
    resultado = comparar(args.tabela)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
