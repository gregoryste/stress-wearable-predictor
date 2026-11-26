"""Prepara a base Stress-Predict no mesmo formato das outras bases.

Base: "Stress-Predict Dataset" (Iqbal et al., 2022, Sensors 22(21):8135,
https://github.com/italha-d/Stress-Predict-Dataset, licença MIT).
Mesma pulseira do WESAD (Empatica E4), mesmas frequências e unidades.

Protocolo (cerca de 1 hora por pessoa): linha de base com questionários, teste de
Stroop, descanso, entrevista, descanso, hiperventilação e descanso final.

Rótulos: o arquivo processado dos autores (Improved_All_Combined_hr_rsp_binary.csv)
marca cada segundo como estresse (1) ou não (0). Usamos esse arquivo, e não os
toques no botão, porque os toques estão errados em alguns participantes (S06, S17,
S18, S30) e os autores corrigiram os rótulos a partir da planilha de horários.
Cada participante tem sete trechos, na ordem: início, Stroop, descanso, entrevista,
descanso, hiperventilação e final.

Calibração (padrão pessoal): os 5 minutos imediatamente antes do Stroop. O começo
da sessão fica de fora da calibração porque inclui o termo de consentimento e a
colocação da pulseira; ele entra como repouso comum.

Fica de fora o S01, que não tem rótulos no arquivo dos autores.

Uso (na raiz do repositório, com o repositório da base clonado em data/):
    python -m treino.preparar_stress_predict --base data/stress_predict/repo
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from shared.features import DURACAO_JANELA_S, extrair_features
from treino.preparar_physionet import ARQUIVOS, COLUNAS, janelas

ARQUIVO_ROTULOS = Path("Processed_data/Improved_All_Combined_hr_rsp_binary.csv")
CALIBRACAO_S = 300
# Trechos de rótulo esperados, na ordem do protocolo: (fase, estado).
TRECHOS = [
    ("Inicio", "repouso"),
    ("Stroop", "estresse"),
    ("Descanso 1", "repouso"),
    ("Entrevista", "estresse"),
    ("Descanso 2", "repouso"),
    ("Hiperventilacao", "estresse"),
    ("Final", "repouso"),
]


def ler_sinal(caminho: Path) -> tuple[float, float, np.ndarray]:
    """Lê um CSV da E4: 1ª linha = início (timestamp Unix, UTC), 2ª = frequência (Hz)."""
    with open(caminho, encoding="utf-8") as arquivo:
        inicio = float(arquivo.readline().split(",")[0])
        frequencia = float(arquivo.readline().split(",")[0])
    valores = np.loadtxt(caminho, delimiter=",", skiprows=2, ndmin=2)
    return inicio, frequencia, valores


def trechos_rotulados(segundos: np.ndarray, rotulos: np.ndarray) -> list[tuple[float, float, int]]:
    """(início, fim, rótulo) de cada sequência de segundos com o mesmo rótulo."""
    ordem = np.argsort(segundos)
    segundos, rotulos = segundos[ordem], rotulos[ordem]
    cortes = np.flatnonzero(np.diff(rotulos) != 0) + 1
    inicios, fins = np.r_[0, cortes], np.r_[cortes, len(rotulos)]
    return [(float(segundos[a]), float(segundos[b - 1]) + 1, int(rotulos[a])) for a, b in zip(inicios, fins)]


def fases_do_participante(trechos: list[tuple[float, float, int]]) -> list[tuple[str, float, float, str]]:
    """(fase, início, fim, estado) a partir dos sete trechos, com a calibração separada.

    Levanta ``ValueError`` se os trechos não seguirem o protocolo.
    """
    esperado = [1 if estado == "estresse" else 0 for _, estado in TRECHOS]
    if [rotulo for _, _, rotulo in trechos] != esperado:
        raise ValueError(f"trechos fora do protocolo: {[r for _, _, r in trechos]}")
    fases = []
    for (nome, estado), (a, b, _) in zip(TRECHOS, trechos):
        if nome == "Inicio":
            corte = max(a, b - CALIBRACAO_S)
            if corte > a:
                fases.append(("Inicio", a, corte, "repouso"))
            fases.append(("Linha de base", corte, b, "repouso"))
        else:
            fases.append((nome, a, b, estado))
    return fases


def preparar(base: Path, saida: Path) -> dict:
    rotulos = pd.read_csv(base / ARQUIVO_ROTULOS)
    saida.parent.mkdir(parents=True, exist_ok=True)
    contagem, descartes, pulados = Counter(), Counter(), []
    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        for pasta in sorted(p for p in (base / "Raw_data").iterdir() if p.is_dir()):
            participante = pasta.name
            numero = int(participante.lstrip("S"))
            do_participante = rotulos[rotulos.Participant == numero]
            if do_participante.empty:
                pulados.append(f"{participante} (sem rótulos)")
                continue

            sinais, inicio = {}, None
            for nome, arquivo_sinal in ARQUIVOS.items():
                comeco, frequencia, valores = ler_sinal(pasta / arquivo_sinal)
                inicio = comeco if inicio is None else inicio
                if comeco != inicio:
                    raise ValueError(f"{pasta}: {arquivo_sinal} começa em outro horário")
                sinais[nome] = (frequencia, valores)

            trechos = trechos_rotulados(
                do_participante["Time(sec)"].to_numpy(dtype=float) - inicio,
                do_participante.Label.to_numpy(dtype=int),
            )
            try:
                fases = fases_do_participante(trechos)
            except ValueError as erro:
                pulados.append(f"{participante} ({erro})")
                continue

            for fase, a, b, estado in fases:
                for inicio_janela, recorte in janelas(sinais, a, b):
                    try:
                        features = extrair_features(recorte)
                    except ValueError as erro:
                        descartes[str(erro).split(":")[0]] += 1
                        continue
                    escritor.writerow({
                        "participante": f"sp-{participante}",
                        "base": "stress_predict",
                        "sessao": "STRESS",
                        "fase": fase,
                        "inicio_s": round(inicio_janela, 1),
                        "fim_s": round(inicio_janela + DURACAO_JANELA_S, 1),
                        "estado": estado,
                        "calibracao": fase == "Linha de base",
                        **features.model_dump(),
                    })
                    contagem[estado] += 1
    return {"janelas_por_estado": dict(contagem), "descartadas": dict(descartes), "pulados": pulados}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=Path("data/stress_predict/repo"),
                        help="pasta do repositório Stress-Predict-Dataset")
    parser.add_argument("--saida", type=Path, default=Path("data/features_stress_predict.csv"))
    args = parser.parse_args()
    resumo = preparar(args.base, args.saida)
    print(f"Janelas: {resumo['janelas_por_estado']} | descartadas: {resumo['descartadas'] or 0}")
    print(f"Fora: {', '.join(resumo['pulados']) or 'ninguém'}")
    print(f"Tabela: {args.saida}")


if __name__ == "__main__":
    main()
