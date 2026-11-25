"""Treina e avalia o modelo de estado (repouso, estresse, diversão).

Lê a tabela gerada por ``treino.preparar_wesad`` e:

1. separa, para cada participante, os primeiros 5 minutos de repouso como
   calibração (o "padrão pessoal" de cada um);
2. avalia duas variantes com leave-one-subject-out (treina com 14 pessoas, testa
   na que ficou de fora):
   - geral: features como medidas;
   - pessoal: cada feature relativa ao repouso da própria pessoa;
3. analisa a previsão da próxima janela (60 s à frente), comparando com a regra
   "o próximo minuto será igual ao atual";
4. treina os modelos finais com todos os participantes e salva:
   - modelos/modelo_estresse.joblib e modelos/metadados.json (para a API);
   - resultados/metricas.json e gráficos PNG (para o relatório).

Uso (na raiz do repositório, depois de preparar o WESAD):
    python -m treino.treinar
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from contratos.api import Features
from shared.features import DURACAO_JANELA_S

FEATURES = list(Features.model_fields)
CLASSES = ["repouso", "estresse", "diversao"]
CALIBRACAO_S = 300  # 5 minutos de repouso no início de cada participante
SEMENTE = 42
N_ARVORES = 300
FOLHA_MINIMA = 3

# Referência: Schmidt et al. (2018), três classes, só pulso, leave-one-subject-out,
# conforme reproduzido em Hagos et al. (2026), tabela VIII.
REFERENCIA = {
    "fonte": "Schmidt et al. (2018), três classes, só pulso, LOSO",
    "random_forest": {"f1_macro": 62.86, "acuracia": 74.85},
    "adaboost": {"f1_macro": 64.12, "acuracia": 75.21},
}


# --- calibração e padrão pessoal ---------------------------------------------------


def marcar_calibracao(df: pd.DataFrame, calibracao_s: int = CALIBRACAO_S) -> tuple[pd.Series, pd.Series]:
    """Marca as janelas de calibração e as que se sobrepõem a ela.

    Calibração: janelas de repouso que terminam nos primeiros ``calibracao_s``
    segundos de repouso do participante. Sobrepostas: janelas que começam antes do
    fim da calibração sem fazer parte dela (compartilham sinal com a calibração e
    por isso ficam fora da avaliação).
    """
    inicio_repouso = df[df.estado == "repouso"].groupby("participante").inicio_s.min()
    fim_calibracao = df.participante.map(inicio_repouso) + calibracao_s
    calibracao = (df.estado == "repouso") & (df.fim_s <= fim_calibracao)
    sobreposta = (df.inicio_s < fim_calibracao) & ~calibracao
    return calibracao, sobreposta


def estatisticas_calibracao(df: pd.DataFrame, calibracao: pd.Series) -> dict:
    """Média e desvio de cada feature no repouso de calibração de cada participante."""
    estatisticas = {}
    for participante, grupo in df[calibracao].groupby("participante"):
        estatisticas[participante] = {
            feature: {
                "media": float(grupo[feature].mean()),
                "desvio": float(grupo[feature].std(ddof=0)) or 1.0,
            }
            for feature in FEATURES
        }
    return estatisticas


def normalizar(df: pd.DataFrame, estatisticas: dict) -> pd.DataFrame:
    """Features relativas ao repouso de cada participante: valor - média da calibração.

    Só a diferença, sem dividir pelo desvio: com 5 minutos de calibração em janelas
    sobrepostas, o desvio fica pequeno e instável, e a divisão piorava o modelo
    (F1 69 contra 81 na validação leave-one-subject-out).
    """
    normalizado = df[FEATURES].astype(float).copy()
    for participante, indices in df.groupby("participante").groups.items():
        for feature in FEATURES:
            ref = estatisticas[participante][feature]
            normalizado.loc[indices, feature] = df.loc[indices, feature] - ref["media"]
    return normalizado


# --- avaliação -----------------------------------------------------------------------


def novo_modelo(n_arvores: int = N_ARVORES) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_arvores,
        min_samples_leaf=FOLHA_MINIMA,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEMENTE,
    )


def metricas(verdadeiro, previsto, participantes) -> dict:
    """Métricas de três classes, binárias (estresse x resto) e por participante."""
    verdadeiro, previsto = np.asarray(verdadeiro), np.asarray(previsto)
    participantes = np.asarray(participantes)
    estresse_v, estresse_p = verdadeiro == "estresse", previsto == "estresse"
    return {
        "janelas": int(len(verdadeiro)),
        "f1_macro": round(100 * f1_score(verdadeiro, previsto, average="macro", labels=CLASSES), 2),
        "acuracia": round(100 * accuracy_score(verdadeiro, previsto), 2),
        "f1_por_classe": {
            classe: round(100 * float(valor), 2)
            for classe, valor in zip(
                CLASSES, f1_score(verdadeiro, previsto, average=None, labels=CLASSES, zero_division=0)
            )
        },
        "binario_estresse": {
            "f1_estresse": round(100 * f1_score(estresse_v, estresse_p, zero_division=0), 2),
            "acuracia": round(100 * accuracy_score(estresse_v, estresse_p), 2),
        },
        "matriz_confusao": {
            "ordem": CLASSES,
            "valores": confusion_matrix(verdadeiro, previsto, labels=CLASSES).tolist(),
        },
        "f1_por_participante": {
            p: round(
                100 * f1_score(verdadeiro[participantes == p], previsto[participantes == p],
                               average="macro", labels=CLASSES, zero_division=0),
                2,
            )
            for p in sorted(set(participantes), key=lambda nome: int(nome[1:]))
        },
    }


def avaliar_loso(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, n_arvores: int) -> np.ndarray:
    """Previsões leave-one-subject-out: cada janela é prevista por um modelo que não viu a pessoa."""
    return cross_val_predict(novo_modelo(n_arvores), X, y, groups=grupos, cv=LeaveOneGroupOut())


def pares_proxima_janela(df: pd.DataFrame) -> pd.DataFrame:
    """Liga cada janela à janela do mesmo participante que começa 60 s depois."""
    seguinte = df[["participante", "inicio_s", "estado"]].rename(
        columns={"inicio_s": "inicio_seguinte", "estado": "estado_seguinte"}
    )
    atual = df.assign(inicio_seguinte=df.inicio_s + DURACAO_JANELA_S).reset_index()
    pares = atual.merge(seguinte, on=["participante", "inicio_seguinte"], how="inner")
    return pares.set_index("index")


# --- gráficos para o relatório -----------------------------------------------------------

AZUL, LARANJA = "#2a78d6", "#eb6834"  # slots 1 e 2 da paleta de referência (validados)
TEXTO, TEXTO_SECUNDARIO, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NOMES = {"repouso": "Repouso", "estresse": "Estresse", "diversao": "Diversão"}


def _estilo(ax):
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO_SECUNDARIO, labelsize=9)


def gerar_graficos(resultado: dict, importancias: dict, pasta: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 10, "text.color": TEXTO, "axes.labelcolor": TEXTO_SECUNDARIO})
    pasta.mkdir(parents=True, exist_ok=True)
    arquivos = []

    # 1. Matriz de confusão do modelo pessoal (proporção por linha).
    pessoal = resultado["avaliacao"]["pessoal"]
    valores = np.array(pessoal["matriz_confusao"]["valores"], dtype=float)
    proporcao = valores / valores.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=FUNDO)
    ax.imshow(proporcao, cmap=LinearSegmentedColormap.from_list("azul", RAMPA_AZUL), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            cor = FUNDO if proporcao[i, j] > 0.55 else TEXTO
            ax.text(j, i, f"{100 * proporcao[i, j]:.0f}%\n({int(valores[i, j])})",
                    ha="center", va="center", color=cor, fontsize=10)
    rotulos = [NOMES[c] for c in CLASSES]
    ax.set_xticks(range(3), rotulos)
    ax.set_yticks(range(3), rotulos)
    ax.set_xlabel("Estado previsto")
    ax.set_ylabel("Estado real")
    ax.tick_params(colors=TEXTO_SECUNDARIO, length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title(
        f"Modelo com padrão pessoal: F1 {pessoal['f1_macro']:.1f}, acurácia {pessoal['acuracia']:.1f}%",
        loc="left", fontsize=11, color=TEXTO,
    )
    fig.tight_layout()
    arquivos.append(pasta / "matriz_confusao.png")
    fig.savefig(arquivos[-1], dpi=160, facecolor=FUNDO)
    plt.close(fig)

    # 2. F1 por participante: geral x pessoal (pontos ligados, um par por pessoa).
    geral_p = resultado["avaliacao"]["geral"]["f1_por_participante"]
    pessoal_p = pessoal["f1_por_participante"]
    participantes = list(geral_p)
    x = np.arange(len(participantes))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), facecolor=FUNDO)
    _estilo(ax)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, p in enumerate(participantes):
        ax.plot([i, i], [geral_p[p], pessoal_p[p]], color=GRADE, linewidth=2, zorder=1)
    ax.scatter(x, [geral_p[p] for p in participantes], s=46, color=AZUL, zorder=2,
               edgecolor=FUNDO, linewidth=2, label="Geral (features como medidas)")
    ax.scatter(x, [pessoal_p[p] for p in participantes], s=46, color=LARANJA, zorder=3,
               edgecolor=FUNDO, linewidth=2, label="Pessoal (relativo ao repouso da pessoa)")
    ax.set_xticks(x, participantes)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("F1 macro (%)")
    ax.set_title("F1 de cada participante quando ele fica de fora do treino",
                 loc="left", fontsize=11, color=TEXTO)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXTO_SECUNDARIO)
    fig.tight_layout()
    arquivos.append(pasta / "f1_por_participante.png")
    fig.savefig(arquivos[-1], dpi=160, facecolor=FUNDO)
    plt.close(fig)

    # 3. Importância das features no modelo pessoal (uma série, barras ordenadas).
    ordenadas = sorted(importancias.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=FUNDO)
    _estilo(ax)
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.barh([nome for nome, _ in ordenadas], [100 * v for _, v in ordenadas], color=AZUL, height=0.6)
    for i, (_, valor) in enumerate(ordenadas):
        ax.text(100 * valor + 0.4, i, f"{100 * valor:.1f}%", va="center", fontsize=8.5, color=TEXTO_SECUNDARIO)
    ax.set_xlabel("Importância no modelo (%)")
    ax.set_title("Quanto cada feature pesa nas decisões do modelo pessoal",
                 loc="left", fontsize=11, color=TEXTO)
    fig.tight_layout()
    arquivos.append(pasta / "importancia_features.png")
    fig.savefig(arquivos[-1], dpi=160, facecolor=FUNDO)
    plt.close(fig)

    return arquivos


# --- pipeline -----------------------------------------------------------------------------


def treinar(tabela: Path, pasta_modelos: Path, pasta_resultados: Path,
            n_arvores: int = N_ARVORES, graficos: bool = True) -> dict:
    df = pd.read_csv(tabela)
    calibracao, sobreposta = marcar_calibracao(df)
    estatisticas = estatisticas_calibracao(df, calibracao)
    avaliacao = df[~calibracao & ~sobreposta]
    grupos, y = avaliacao.participante, avaliacao.estado

    # 1 e 2. Avaliação leave-one-subject-out das duas variantes, nas mesmas janelas.
    X_geral = avaliacao[FEATURES]
    X_pessoal = normalizar(avaliacao, estatisticas)
    previsto_geral = avaliar_loso(X_geral, y, grupos, n_arvores)
    previsto_pessoal = avaliar_loso(X_pessoal, y, grupos, n_arvores)

    # 3. Próxima janela (60 s à frente).
    pares = pares_proxima_janela(avaliacao)
    previsto_atual = pd.Series(previsto_pessoal, index=avaliacao.index)
    modelo_proxima = avaliar_loso(
        X_pessoal.loc[pares.index], pares.estado_seguinte, pares.participante, n_arvores
    ) if len(pares.participante.unique()) > 1 else np.asarray(pares.estado)
    proxima = {
        "pares": int(len(pares)),
        "pares_com_mudanca_de_estado": int((pares.estado != pares.estado_seguinte).sum()),
        "regra_repetir_estado_real": metricas(pares.estado_seguinte, pares.estado, pares.participante),
        "regra_repetir_estado_previsto": metricas(
            pares.estado_seguinte, previsto_atual.loc[pares.index], pares.participante
        ),
        "modelo_dedicado": metricas(pares.estado_seguinte, modelo_proxima, pares.participante),
    }
    for chave in ("regra_repetir_estado_real", "regra_repetir_estado_previsto", "modelo_dedicado"):
        proxima[chave] = {k: proxima[chave][k] for k in ("janelas", "f1_macro", "acuracia")}

    # 4. Modelos finais, com todas as janelas de todos os participantes.
    # Treinados com arrays (na ordem de FEATURES), para a API prever sem precisar do pandas.
    modelo_geral = novo_modelo(n_arvores).fit(df[FEATURES].to_numpy(), df.estado.to_numpy())
    modelo_pessoal = novo_modelo(n_arvores).fit(
        normalizar(df, estatisticas).to_numpy(), df.estado.to_numpy()
    )
    importancias = dict(zip(FEATURES, modelo_pessoal.feature_importances_.round(4).tolist()))

    agora = datetime.now(timezone.utc)
    versao = f"rf-v1-{agora:%Y%m%d}"
    resultado = {
        "versao_modelo": versao,
        "gerado_em": agora.isoformat(timespec="seconds"),
        "dados": {
            "tabela": str(tabela),
            "participantes": int(df.participante.nunique()),
            "janelas_total": int(len(df)),
            "janelas_calibracao": int(calibracao.sum()),
            "janelas_fora_por_sobreposicao": int(sobreposta.sum()),
            "janelas_avaliadas": int(len(avaliacao)),
            "calibracao_s": CALIBRACAO_S,
        },
        "validacao": "leave-one-subject-out (15 rodadas)",
        "referencia": REFERENCIA,
        "avaliacao": {
            "geral": metricas(y, previsto_geral, grupos),
            "pessoal": metricas(y, previsto_pessoal, grupos),
        },
        "proxima_janela": proxima,
        "importancia_features_pessoal": importancias,
        "importancia_tipo": "redução média de impureza (MDI) do Random Forest",
    }

    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    arquivo_modelo = pasta_modelos / "modelo_estresse.joblib"
    joblib.dump(
        {
            "versao": versao,
            "features": FEATURES,
            "classes": list(modelo_pessoal.classes_),
            "geral": modelo_geral,
            "pessoal": modelo_pessoal,
            "calibracao": estatisticas,
        },
        arquivo_modelo,
        compress=3,
    )
    metadados = {
        "versao": versao,
        "criado_em": resultado["gerado_em"],
        "arquivo": arquivo_modelo.name,
        "features": FEATURES,
        "classes": list(modelo_pessoal.classes_),
        "variantes": {
            "geral": "features como medidas; serve para qualquer usuário",
            "pessoal": "feature menos a média do repouso de calibração do usuário; exige calibração",
        },
        "usuarios_calibrados": sorted(estatisticas, key=lambda nome: int(nome[1:])),
        "parametros": {"modelo": "RandomForestClassifier", "n_arvores": n_arvores,
                       "folha_minima": FOLHA_MINIMA, "class_weight": "balanced", "semente": SEMENTE,
                       "janela_s": DURACAO_JANELA_S, "calibracao_s": CALIBRACAO_S},
        "metricas_loso": {v: {k: resultado["avaliacao"][v][k] for k in ("f1_macro", "acuracia")}
                          for v in ("geral", "pessoal")},
        "versoes": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
    }
    (pasta_modelos / "metadados.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (pasta_resultados / "metricas.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if graficos:
        resultado["graficos"] = [str(p) for p in gerar_graficos(resultado, importancias, pasta_resultados)]
    resultado["tamanho_modelo_mb"] = round(arquivo_modelo.stat().st_size / 1e6, 2)
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tabela", type=Path, default=Path("data/features_wesad.csv"))
    parser.add_argument("--modelos", type=Path, default=Path("modelos"))
    parser.add_argument("--resultados", type=Path, default=Path("resultados"))
    parser.add_argument("--arvores", type=int, default=N_ARVORES)
    args = parser.parse_args()
    if not args.tabela.exists():
        parser.error(f"{args.tabela} não existe. Rode antes: python -m treino.preparar_wesad")

    r = treinar(args.tabela, args.modelos, args.resultados, args.arvores)
    ref = REFERENCIA["adaboost"]
    print(f"Modelo {r['versao_modelo']} ({r['tamanho_modelo_mb']} MB), "
          f"{r['dados']['janelas_avaliadas']} janelas avaliadas, {r['validacao']}")
    print(f"  Referência (artigo, só pulso): F1 {ref['f1_macro']}, acurácia {ref['acuracia']}")
    for variante in ("geral", "pessoal"):
        a = r["avaliacao"][variante]
        print(f"  {variante:8s}: F1 {a['f1_macro']}, acurácia {a['acuracia']}, "
              f"F1 por classe {a['f1_por_classe']}, estresse x resto F1 {a['binario_estresse']['f1_estresse']}")
    p = r["proxima_janela"]
    print(f"  Próxima janela: {p['pares']} pares, {p['pares_com_mudanca_de_estado']} com mudança de estado")
    for nome in ("regra_repetir_estado_real", "regra_repetir_estado_previsto", "modelo_dedicado"):
        print(f"    {nome}: F1 {p[nome]['f1_macro']}, acurácia {p[nome]['acuracia']}")
    print(f"Arquivos: {args.modelos}/, {args.resultados}/")


if __name__ == "__main__":
    main()
