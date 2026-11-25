"""Prepara a base do PhysioNet (estresse + exercício) no mesmo formato do WESAD.

Base: "Wearable Device Dataset from Induced Stress and Structured Exercise Sessions"
(Hongn et al., 2025, PhysioNet, DOI 10.13026/he0v-tf17, licença ODC-By 1.0).
Mesma pulseira do WESAD (Empatica E4), mesmas frequências e unidades.

Rótulos, a partir dos toques no botão da pulseira (tags.csv), com o mapeamento do
notebook dos autores (posição i = i-ésima linha do tags.csv):
- repouso: linha de base e os dois descansos;
- estresse: Stroop, TMCT e o bloco das tarefas de opinião (e contas, na versão 1),
  tratado como contínuo porque cada tarefa dura só 30 a 40 s, menos que uma janela;
- exercicio: sessões de bicicleta (aeróbica e anaeróbica), do primeiro ao último toque.

A linha de base de cada participante é a calibração do padrão pessoal.

Ficam de fora, conforme data_constraints.txt dos autores: S02 (sinais duplicados),
f07 (sensor de pulso e temperatura coberto), f14 (sessão dividida) e as sessões de
exercício divididas ou incompletas em partes (S11_a/b, S16_a/b).

Uso (na raiz do repositório):
    python -m treino.preparar_physionet --base data/physionet_wearable/extraido/<pasta>/Wearable_Dataset
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from contratos.api import Features
from shared.features import DURACAO_JANELA_S, extrair_features

# (fase, linha de início, linha de fim, estado). Linhas do tags.csv, contando de 1.
FASES_ESTRESSE = {
    "v1": [
        ("Baseline", 1, 2, "repouso"),
        ("Stroop", 3, 4, "estresse"),
        ("First Rest", 4, 5, "repouso"),
        ("TMCT", 5, 6, "estresse"),
        ("Second Rest", 6, 7, "repouso"),
        ("Opinioes e contas", 7, 12, "estresse"),
    ],
    "v2": [
        ("Baseline", 1, 2, "repouso"),
        ("TMCT", 2, 3, "estresse"),
        ("First Rest", 3, 4, "repouso"),
        ("Opinioes", 4, 7, "estresse"),
        ("Second Rest", 7, 8, "repouso"),
    ],
}
EXCLUIDOS = {"STRESS": {"S02", "f07", "f14_a", "f14_b"}}
ARQUIVOS = {"bvp": "BVP.csv", "eda": "EDA.csv", "temp": "TEMP.csv", "acc": "ACC.csv"}
PASSO_S = 10
COLUNAS = ["participante", "base", "sessao", "fase", "inicio_s", "fim_s", "estado", "calibracao"] + list(
    Features.model_fields
)


def ler_sinal(caminho: Path) -> tuple[pd.Timestamp, float, np.ndarray]:
    """Lê um CSV da E4: 1ª linha = início (UTC), 2ª = frequência (Hz), depois os valores."""
    with open(caminho, encoding="utf-8") as arquivo:
        inicio = pd.Timestamp(arquivo.readline().split(",")[0].strip())
        frequencia = float(arquivo.readline().split(",")[0])
    valores = np.loadtxt(caminho, delimiter=",", skiprows=2, ndmin=2)
    return inicio, frequencia, valores


def ler_toques(pasta: Path, inicio: pd.Timestamp) -> np.ndarray:
    """Segundos de cada toque no botão, desde o início da sessão."""
    caminho = pasta / "tags.csv"
    if caminho.stat().st_size == 0:
        return np.empty(0)
    toques = pd.to_datetime(pd.read_csv(caminho, header=None)[0])
    return (toques - inicio).dt.total_seconds().to_numpy()


def fases_da_sessao(sessao: str, participante: str, toques: np.ndarray) -> list[tuple[str, float, float, str]]:
    """(fase, início em s, fim em s, estado) de cada trecho rotulado da sessão."""
    if sessao == "STRESS":
        versao = "v1" if participante.startswith("S") else "v2"
        return [
            (nome, toques[a - 1], toques[b - 1], estado)
            for nome, a, b, estado in FASES_ESTRESSE[versao]
            if b <= len(toques)
        ]
    if len(toques) >= 2:  # AEROBIC / ANAEROBIC: do primeiro ao último toque
        return [(sessao.capitalize(), toques[0], toques[-1], "exercicio")]
    return []


def janelas(sinais: dict, inicio_fase: float, fim_fase: float):
    """Janelas de 60 s, a cada PASSO_S, inteiras dentro da fase."""
    for inicio in np.arange(np.ceil(inicio_fase), fim_fase - DURACAO_JANELA_S + 1e-9, PASSO_S):
        recorte = {}
        for nome, (frequencia, valores) in sinais.items():
            a = int(round(inicio * frequencia))
            b = a + int(round(DURACAO_JANELA_S * frequencia))
            recorte[nome] = valores[a:b] if nome == "acc" else valores[a:b, 0]
        yield float(inicio), recorte


def preparar(base: Path, saida: Path) -> dict:
    saida.parent.mkdir(parents=True, exist_ok=True)
    contagem, descartes, pulados = Counter(), Counter(), []
    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        for sessao in ("STRESS", "AEROBIC", "ANAEROBIC"):
            for pasta in sorted((base / sessao).iterdir()):
                participante = pasta.name
                if participante in EXCLUIDOS.get(sessao, set()) or "_" in participante:
                    pulados.append(f"{sessao}/{participante}")
                    continue
                sinais, inicio = {}, None
                for nome, arquivo_sinal in ARQUIVOS.items():
                    comeco, frequencia, valores = ler_sinal(pasta / arquivo_sinal)
                    inicio = inicio or comeco
                    if comeco != inicio:
                        raise ValueError(f"{pasta}: {arquivo_sinal} começa em outro horário")
                    sinais[nome] = (frequencia, valores)
                for fase, a, b, estado in fases_da_sessao(sessao, participante, ler_toques(pasta, inicio)):
                    for inicio_janela, recorte in janelas(sinais, a, b):
                        try:
                            features = extrair_features(recorte)
                        except ValueError as erro:
                            descartes[str(erro).split(":")[0]] += 1
                            continue
                        escritor.writerow({
                            "participante": f"pn-{participante}",
                            "base": "physionet",
                            "sessao": sessao,
                            "fase": fase,
                            "inicio_s": round(inicio_janela, 1),
                            "fim_s": round(inicio_janela + DURACAO_JANELA_S, 1),
                            "estado": estado,
                            "calibracao": sessao == "STRESS" and fase == "Baseline",
                            **features.model_dump(),
                        })
                        contagem[estado] += 1
    return {"janelas_por_estado": dict(contagem), "descartadas": dict(descartes), "pulados": pulados}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="pasta Wearable_Dataset da base")
    parser.add_argument("--saida", type=Path, default=Path("data/features_physionet.csv"))
    args = parser.parse_args()
    resumo = preparar(args.base, args.saida)
    print(f"Janelas: {resumo['janelas_por_estado']} | descartadas: {resumo['descartadas'] or 0}")
    print(f"Fora: {', '.join(resumo['pulados'])}")
    print(f"Tabela: {args.saida}")


if __name__ == "__main__":
    main()
