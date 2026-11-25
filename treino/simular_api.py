"""Simula o uso real do sistema: reproduz a sessão de cada participante pela API.

Para cada participante do WESAD:
1. treina um modelo SEM esse participante (como no leave-one-subject-out); dele,
   só entram os 5 minutos de repouso de calibração;
2. corta a sessão inteira em janelas de 60 s seguidas, como a ingestão fará ao vivo,
   calcula as features e chama ``POST /prever`` da API, em ordem;
3. mede o que importa para quem usa: acerto por janela, tempo até o primeiro
   alerta depois que o estresse começa, alertas falsos e tempo de resposta.

Gera resultados/simulacao_api.json e resultados/linha_do_tempo_S2.png.

Uso (na raiz, depois de preparar o WESAD):
    python -m treino.simular_api
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api_ia.main import app, obter_servico
from api_ia.modelo import ServicoModelo
from contratos.api import JanelaFeatures
from shared.features import DURACAO_JANELA_S, extrair_features
from treino.preparar_wesad import CONDICOES, FREQUENCIA_ROTULOS_HZ, SINAIS_WESAD, carregar_participante
from treino.treinar import FEATURES, estatisticas_calibracao, marcar_calibracao, normalizar, novo_modelo
from shared.features import FREQUENCIAS_HZ

ALERTAS = {"aviso_antecipado", "respiracao_guiada", "pausa_ativa"}
INICIO_SESSAO = pd.Timestamp("2026-11-27T14:00:00Z")


def servico_sem(participante: str, tabela: pd.DataFrame, pasta: Path, n_arvores: int) -> ServicoModelo:
    """Serviço da API com um modelo treinado sem o participante (mas com a calibração dele)."""
    calibracao, _ = marcar_calibracao(tabela)
    estatisticas = estatisticas_calibracao(tabela, calibracao)
    treino = tabela[tabela.participante != participante]
    pacote = {
        "versao": f"simulacao-sem-{participante}",
        "features": FEATURES,
        "classes": sorted(CONDICOES.values()),
        "geral": novo_modelo(n_arvores).fit(treino[FEATURES].to_numpy(), treino.estado.to_numpy()),
        "pessoal": novo_modelo(n_arvores).fit(
            normalizar(treino, estatisticas).to_numpy(), treino.estado.to_numpy()
        ),
        "calibracao": estatisticas,
    }
    caminho = pasta / f"modelo_sem_{participante}.joblib"
    joblib.dump(pacote, caminho)
    return ServicoModelo(caminho)


def janelas_da_sessao(wrist: dict, rotulos: np.ndarray):
    """Janelas de 60 s seguidas (sem sobreposição) cobrindo a sessão inteira."""
    duracao = min(
        len(rotulos) // FREQUENCIA_ROTULOS_HZ,
        *(len(wrist[nome]) // FREQUENCIAS_HZ[sinal] for sinal, nome in SINAIS_WESAD.items()),
    )
    for inicio in range(0, duracao - DURACAO_JANELA_S + 1, DURACAO_JANELA_S):
        trecho = rotulos[inicio * FREQUENCIA_ROTULOS_HZ : (inicio + DURACAO_JANELA_S) * FREQUENCIA_ROTULOS_HZ]
        rotulo = int(trecho[0])
        condicao = CONDICOES.get(rotulo, "outro") if np.all(trecho == rotulo) else "outro"
        sinais = {
            sinal: wrist[nome][inicio * FREQUENCIAS_HZ[sinal] : (inicio + DURACAO_JANELA_S) * FREQUENCIAS_HZ[sinal]]
            for sinal, nome in SINAIS_WESAD.items()
        }
        sinais = {s: (v if s == "acc" else np.asarray(v).reshape(-1)) for s, v in sinais.items()}
        yield inicio, condicao, sinais


def simular_participante(participante: str, caminho_pkl: Path, servico: ServicoModelo) -> pd.DataFrame:
    """Envia a sessão inteira, minuto a minuto, para a API. Devolve uma linha por janela."""
    app.dependency_overrides[obter_servico] = lambda: servico
    cliente = TestClient(app)
    wrist, rotulos = carregar_participante(caminho_pkl)
    linhas = []
    try:
        for inicio, condicao, sinais in janelas_da_sessao(wrist, rotulos):
            try:
                features = extrair_features(sinais)
            except ValueError:
                linhas.append({"inicio_min": inicio / 60, "condicao": condicao, "descartada": True})
                continue
            entrada = JanelaFeatures(
                usuario=participante,
                fim_janela=INICIO_SESSAO + pd.Timedelta(seconds=inicio + DURACAO_JANELA_S),
                features=features,
            )
            antes = time.perf_counter()
            resposta = cliente.post("/prever", json=entrada.model_dump(mode="json"))
            latencia_ms = 1000 * (time.perf_counter() - antes)
            resposta.raise_for_status()
            corpo = resposta.json()
            linhas.append({
                "inicio_min": inicio / 60,
                "condicao": condicao,
                "descartada": False,
                "estado": corpo["estado"],
                "probabilidade": corpo["probabilidade"],
                "prob_estresse": servico.probabilidades(entrada).get("estresse", 0.0),
                "intervencao": corpo["intervencao"],
                "latencia_ms": latencia_ms,
            })
    finally:
        app.dependency_overrides.clear()
    return pd.DataFrame(linhas)


def resumo_participante(sessao: pd.DataFrame) -> dict:
    validas = sessao[~sessao.descartada]
    rotuladas = validas[validas.condicao.isin(CONDICOES.values())]
    estresse = validas[validas.condicao == "estresse"]
    inicio_estresse = estresse.inicio_min.min()
    alertas_no_estresse = estresse[estresse.intervencao.isin(ALERTAS)]
    calmas = validas[validas.condicao.isin(["repouso", "diversao"])]
    falsos = calmas[calmas.intervencao.isin(ALERTAS)]
    sem_rotulo = validas[validas.condicao == "outro"]
    alertas_sem_rotulo = sem_rotulo[sem_rotulo.intervencao.isin(ALERTAS)]
    return {
        "janelas": int(len(sessao)),
        "janelas_descartadas": int(sessao.descartada.sum()),
        "acerto_janelas_rotuladas": round(100 * float((rotuladas.estado == rotuladas.condicao).mean()), 1),
        "minutos_de_estresse": int(len(estresse)),
        "minutos_ate_primeiro_alerta": (
            float(alertas_no_estresse.inicio_min.min() - inicio_estresse) if len(alertas_no_estresse) else None
        ),
        "estresse_com_alerta_pct": round(100 * len(alertas_no_estresse) / max(1, len(estresse)), 1),
        "minutos_calmos": int(len(calmas)),
        "alertas_falsos": int(len(falsos)),
        "alertas_falsos_por_hora": round(60 * len(falsos) / max(1, len(calmas)), 2),
        "minutos_sem_rotulo": int(len(sem_rotulo)),
        "alertas_sem_rotulo": int(len(alertas_sem_rotulo)),
    }


def grafico_linha_do_tempo(sessao: pd.DataFrame, participante: str, arquivo: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Paleta de referência validada: slots 1-3 (azul, laranja, água) para as condições reais.
    cores = {"repouso": "#2a78d6", "estresse": "#eb6834", "diversao": "#1baf7a"}
    nomes = {"repouso": "Repouso", "estresse": "Estresse", "diversao": "Diversão"}
    texto, secundario, grade, fundo = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    validas = sessao[~sessao.descartada]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=fundo)
    ax.set_facecolor(fundo)
    for _, linha in sessao.iterrows():
        if linha.condicao in cores:
            ax.axvspan(linha.inicio_min, linha.inicio_min + 1, color=cores[linha.condicao], alpha=0.16, lw=0)
    ax.plot(validas.inicio_min + 0.5, validas.prob_estresse, color=texto, linewidth=1.6, zorder=3)
    ax.axhline(0.5, color=secundario, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(validas.inicio_min.max() + 1.2, 0.5, "limiar", va="center", fontsize=8.5, color=secundario)
    for intervencao, marcador in (("aviso_antecipado", "^"), ("respiracao_guiada", "o"), ("pausa_ativa", "s")):
        pontos = validas[validas.intervencao == intervencao]
        ax.scatter(pontos.inicio_min + 0.5, [1.07] * len(pontos), marker=marcador, s=34,
                   color=texto, edgecolor=fundo, linewidth=1, zorder=4, clip_on=False)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, sessao.inicio_min.max() + 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Minutos de sessão", color=secundario)
    ax.set_ylabel("Probabilidade de estresse", color=secundario)
    ax.tick_params(colors=secundario, labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(grade)
    ax.grid(axis="y", color=grade, linewidth=0.8)
    ax.set_axisbelow(True)
    legenda = [Patch(color=cores[c], alpha=0.35, label=f"{nomes[c]} (real)") for c in cores] + [
        Patch(facecolor=fundo, edgecolor=grade, label="Sem rótulo (transições, meditação)"),
        Line2D([], [], color=texto, linewidth=1.6, label="Probabilidade de estresse (API)"),
        Line2D([], [], color=texto, marker="^", linestyle="", markersize=6, label="Aviso antecipado"),
        Line2D([], [], color=texto, marker="o", linestyle="", markersize=6, label="Respiração guiada"),
        Line2D([], [], color=texto, marker="s", linestyle="", markersize=6, label="Pausa ativa"),
    ]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4,
              frameon=False, fontsize=8.5, labelcolor=secundario)
    fig.suptitle(f"{participante}: a API acompanhando a sessão minuto a minuto (modelo sem {participante} no treino)",
                 x=0.01, ha="left", fontsize=11, color=texto)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=160, facecolor=fundo)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wesad", type=Path, default=Path("WESAD"))
    parser.add_argument("--tabela", type=Path, default=Path("data/features_wesad.csv"))
    parser.add_argument("--resultados", type=Path, default=Path("resultados"))
    parser.add_argument("--arvores", type=int, default=300)
    parser.add_argument("--grafico", default="S2", help="participante da linha do tempo")
    args = parser.parse_args()
    if not args.tabela.exists():
        parser.error(f"{args.tabela} não existe. Rode antes: python -m treino.preparar_wesad")

    tabela = pd.read_csv(args.tabela)
    por_participante, latencias, sessoes = {}, [], []
    with tempfile.TemporaryDirectory() as temporaria:
        for pkl in sorted(args.wesad.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:])):
            servico = servico_sem(pkl.stem, tabela, Path(temporaria), args.arvores)
            sessao = simular_participante(pkl.stem, pkl, servico)
            por_participante[pkl.stem] = resumo_participante(sessao)
            latencias.extend(sessao.latencia_ms.dropna().tolist())
            sessoes.append(sessao.assign(participante=pkl.stem))
            if pkl.stem == args.grafico:
                args.resultados.mkdir(parents=True, exist_ok=True)
                grafico_linha_do_tempo(sessao, pkl.stem, args.resultados / f"linha_do_tempo_{pkl.stem}.png")
            r = por_participante[pkl.stem]
            print(f"{pkl.stem}: acerto {r['acerto_janelas_rotuladas']}%, 1º alerta após "
                  f"{r['minutos_ate_primeiro_alerta']} min, {r['alertas_falsos']} alertas falsos")

    tabela_resumo = pd.DataFrame(por_participante).T
    resumo = {
        "descricao": "Sessão inteira de cada participante enviada à API em janelas de 60 s seguidas; "
                     "modelo treinado sem o participante (só a calibração de 5 min dele).",
        "participantes": len(por_participante),
        "janelas_enviadas": int(tabela_resumo.janelas.sum()),
        "janelas_descartadas": int(tabela_resumo.janelas_descartadas.sum()),
        "acerto_medio_janelas_rotuladas": round(float(tabela_resumo.acerto_janelas_rotuladas.mean()), 1),
        "participantes_com_alerta_no_estresse": int(tabela_resumo.minutos_ate_primeiro_alerta.notna().sum()),
        "mediana_minutos_ate_primeiro_alerta": float(
            pd.to_numeric(tabela_resumo.minutos_ate_primeiro_alerta).median()
        ),
        "estresse_com_alerta_pct_medio": round(float(tabela_resumo.estresse_com_alerta_pct.mean()), 1),
        "alertas_falsos_total": int(tabela_resumo.alertas_falsos.sum()),
        "minutos_calmos_total": int(tabela_resumo.minutos_calmos.sum()),
        "alertas_falsos_por_hora": round(
            60 * float(tabela_resumo.alertas_falsos.sum()) / float(tabela_resumo.minutos_calmos.sum()), 2
        ),
        "minutos_sem_rotulo_total": int(tabela_resumo.minutos_sem_rotulo.sum()),
        "alertas_sem_rotulo_total": int(tabela_resumo.alertas_sem_rotulo.sum()),
        "alertas_sem_rotulo_por_hora": round(
            60 * float(tabela_resumo.alertas_sem_rotulo.sum()) / float(tabela_resumo.minutos_sem_rotulo.sum()), 2
        ),
        "obs_sem_rotulo": "Trechos sem rótulo válido (transições, questionários, meditação). Alertas ali "
                          "podem ser ativação real ou alarmes falsos; o WESAD não permite separar.",
        "latencia_ms": {
            "mediana": round(float(np.median(latencias)), 1),
            "p95": round(float(np.percentile(latencias, 95)), 1),
            "medida": "chamada POST /prever em processo (TestClient), sem rede",
        },
        "por_participante": por_participante,
    }
    pd.concat(sessoes).to_csv(args.resultados / "simulacao_api_janelas.csv", index=False, float_format="%.4f")
    (args.resultados / "simulacao_api.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in resumo.items() if k != "por_participante"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
