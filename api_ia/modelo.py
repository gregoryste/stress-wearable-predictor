"""Carrega o modelo treinado e transforma uma janela de features numa predição."""

import os
from pathlib import Path

import joblib
import numpy as np

from api_ia.regras import Historico, Registro, escolher_intervencao, prever_proxima
from contratos.api import Estado, JanelaFeatures, Predicao

CAMINHO_PADRAO = Path(__file__).resolve().parent.parent / "modelos" / "modelo_estresse.joblib"


class ServicoModelo:
    """Modelo de estado + histórico por usuário + regras de intervenção.

    Usuários com calibração (participantes do WESAD) usam o modelo pessoal, que
    compara cada feature com o repouso da própria pessoa; os demais usam o geral.
    """

    def __init__(self, caminho: str | Path | None = None):
        caminho = Path(caminho or os.getenv("MODELO_PATH") or CAMINHO_PADRAO)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho}. Gere com: python -m treino.treinar"
            )
        # joblib usa pickle: só carregar o arquivo gerado pelo nosso treino.
        pacote = joblib.load(caminho)
        self.versao: str = pacote["versao"]
        self.features: list[str] = pacote["features"]
        self._modelos = {"geral": pacote["geral"], "pessoal": pacote["pessoal"]}
        self._repouso = {
            usuario: np.array([estatisticas[f]["media"] for f in self.features])
            for usuario, estatisticas in pacote["calibracao"].items()
        }
        self.historico = Historico()

    @property
    def usuarios_calibrados(self) -> list[str]:
        return sorted(self._repouso)

    def variante(self, usuario: str) -> str:
        return "pessoal" if usuario in self._repouso else "geral"

    def probabilidades(self, janela: JanelaFeatures) -> dict[str, float]:
        """Probabilidade de cada estado na janela atual."""
        valores = np.array([[getattr(janela.features, f) for f in self.features]], dtype=float)
        variante = self.variante(janela.usuario)
        if variante == "pessoal":
            valores = valores - self._repouso[janela.usuario]
        modelo = self._modelos[variante]
        return {str(classe): float(p) for classe, p in zip(modelo.classes_, modelo.predict_proba(valores)[0])}

    def prever(self, janela: JanelaFeatures) -> Predicao:
        probabilidades = self.probabilidades(janela)
        estado = max(probabilidades, key=probabilidades.get)
        registros = self.historico.registrar(
            janela.usuario,
            Registro(janela.fim_janela, estado, probabilidades.get("estresse", 0.0)),
        )
        proxima, prob_proxima = prever_proxima(registros, probabilidades)
        return Predicao(
            usuario=janela.usuario,
            fim_janela=janela.fim_janela,
            estado=Estado(estado),
            probabilidade=round(probabilidades[estado], 4),
            proxima_janela=Estado(proxima),
            prob_proxima=round(prob_proxima, 4),
            intervencao=escolher_intervencao(registros, proxima),
            versao_modelo=self.versao,
        )
