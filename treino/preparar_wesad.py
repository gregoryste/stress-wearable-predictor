"""Prepara o WESAD para o treino: corta janelas de 60 s e calcula as features.

Lê os 15 participantes (um .pkl de cerca de 1 GB por vez), corta janelas que ficam
inteiras dentro de uma condição (repouso, estresse ou diversão) e calcula as
features com ``shared.features.extrair_features`` — a mesma função que a ingestão
usa ao vivo. O resultado é uma tabela pequena em CSV, que o treino lê em segundos.

Uso (na raiz do repositório):
    python -m treino.preparar_wesad
    python -m treino.preparar_wesad --passo 10 --wesad WESAD --saida data/features_wesad.csv
"""

import argparse
import csv
import json
import pickle
import time
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from contratos.api import Features
from shared.features import DURACAO_JANELA_S, FREQUENCIAS_HZ, extrair_features

# Rótulos do WESAD (readme oficial): 1 = repouso, 2 = estresse, 3 = diversão.
# 0 = transição, 4 = meditação e 5 a 7 = ignorar.
CONDICOES = {1: "repouso", 2: "estresse", 3: "diversao"}
FREQUENCIA_ROTULOS_HZ = 700

# Nome de cada sinal no .pkl do WESAD -> nome usado em shared.features.
SINAIS_WESAD = {"bvp": "BVP", "eda": "EDA", "temp": "TEMP", "acc": "ACC"}

COLUNAS_ID = ["participante", "inicio_s", "fim_s", "rotulo", "estado"]
COLUNAS_FEATURES = list(Features.model_fields)


def carregar_participante(caminho_pkl: Path) -> tuple[dict, np.ndarray]:
    """Lê um .pkl do WESAD e devolve (sinais do pulso, rótulos a 700 Hz)."""
    with open(caminho_pkl, "rb") as arquivo:
        dados = pickle.load(arquivo, encoding="latin1")
    return dados["signal"]["wrist"], np.asarray(dados["label"]).reshape(-1)


def janelas_do_participante(
    wrist: dict, rotulos: np.ndarray, passo_s: int
) -> Iterator[tuple[int, int, dict]]:
    """Gera (início em s, rótulo, sinais) para cada janela inteira numa condição válida."""
    duracao_s = min(
        len(rotulos) // FREQUENCIA_ROTULOS_HZ,
        *(len(wrist[nome]) // FREQUENCIAS_HZ[sinal] for sinal, nome in SINAIS_WESAD.items()),
    )
    for inicio in range(0, duracao_s - DURACAO_JANELA_S + 1, passo_s):
        trecho = rotulos[
            inicio * FREQUENCIA_ROTULOS_HZ : (inicio + DURACAO_JANELA_S) * FREQUENCIA_ROTULOS_HZ
        ]
        rotulo = int(trecho[0])
        if rotulo not in CONDICOES or not np.all(trecho == rotulo):
            continue
        sinais = {}
        for sinal, nome in SINAIS_WESAD.items():
            fs = FREQUENCIAS_HZ[sinal]
            valores = wrist[nome][inicio * fs : (inicio + DURACAO_JANELA_S) * fs]
            sinais[sinal] = valores if sinal == "acc" else np.asarray(valores).reshape(-1)
        yield inicio, rotulo, sinais


def preparar(pasta_wesad: Path, saida: Path, passo_s: int) -> dict:
    """Processa todos os participantes e grava a tabela de features. Devolve um resumo."""
    pkls = sorted(pasta_wesad.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    if not pkls:
        raise FileNotFoundError(f"Nenhum S*/S*.pkl encontrado em {pasta_wesad}")

    saida.parent.mkdir(parents=True, exist_ok=True)
    descartes: Counter = Counter()
    por_estado: Counter = Counter()
    inicio_total = time.perf_counter()

    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_ID + COLUNAS_FEATURES)
        escritor.writeheader()
        for pkl in pkls:
            inicio = time.perf_counter()
            wrist, rotulos = carregar_participante(pkl)
            linhas = 0
            for inicio_s, rotulo, sinais in janelas_do_participante(wrist, rotulos, passo_s):
                try:
                    features = extrair_features(sinais)
                except ValueError as erro:
                    descartes[str(erro).split(":")[0]] += 1
                    continue
                escritor.writerow(
                    {
                        "participante": pkl.stem,
                        "inicio_s": inicio_s,
                        "fim_s": inicio_s + DURACAO_JANELA_S,
                        "rotulo": rotulo,
                        "estado": CONDICOES[rotulo],
                        **features.model_dump(),
                    }
                )
                por_estado[CONDICOES[rotulo]] += 1
                linhas += 1
            print(f"{pkl.stem}: {linhas} janelas em {time.perf_counter() - inicio:.1f} s")
            del wrist, rotulos

    resumo = {
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arquivo": str(saida),
        "participantes": len(pkls),
        "duracao_janela_s": DURACAO_JANELA_S,
        "passo_s": passo_s,
        "janelas": sum(por_estado.values()),
        "janelas_por_estado": dict(por_estado),
        "janelas_descartadas": dict(descartes),
        "tempo_s": round(time.perf_counter() - inicio_total, 1),
    }
    saida.with_suffix(".resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wesad", type=Path, default=Path("WESAD"), help="pasta do WESAD")
    parser.add_argument(
        "--saida", type=Path, default=Path("data/features_wesad.csv"), help="arquivo CSV de saída"
    )
    parser.add_argument(
        "--passo", type=int, default=10, help="distância entre o início de janelas seguidas (s)"
    )
    args = parser.parse_args()
    if args.passo <= 0:
        parser.error("--passo precisa ser positivo")

    resumo = preparar(args.wesad, args.saida, args.passo)
    print(
        f"\n{resumo['janelas']} janelas {resumo['janelas_por_estado']}, "
        f"descartadas: {resumo['janelas_descartadas'] or 0}, em {resumo['tempo_s']} s"
    )
    print(f"Tabela: {args.saida}  |  Resumo: {args.saida.with_suffix('.resumo.json')}")


if __name__ == "__main__":
    main()
