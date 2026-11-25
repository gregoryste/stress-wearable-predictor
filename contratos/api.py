"""Contrato da API de IA.

Define o formato de entrada e de saída do ``POST /prever``. A ingestão (Jackson)
envia uma ``JanelaFeatures`` a cada janela de 60 s e recebe uma ``Predicao``.

Mudar este arquivo é mudar o contrato: só com o acordo dos três.
"""

from enum import Enum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class Estado(str, Enum):
    """Estados que o modelo reconhece (rótulos 1, 2 e 3 do WESAD)."""

    repouso = "repouso"
    estresse = "estresse"
    diversao = "diversao"


class Intervencao(str, Enum):
    """Intervenção sugerida ao usuário, da mais leve à mais forte."""

    nenhuma = "nenhuma"
    aviso_antecipado = "aviso_antecipado"
    respiracao_guiada = "respiracao_guiada"
    pausa_ativa = "pausa_ativa"
    ajuste_ambiente = "ajuste_ambiente"
    encaminhamento = "encaminhamento"


class Features(BaseModel):
    """Features extraídas de uma janela de sinais do pulso (Empatica E4)."""

    model_config = ConfigDict(extra="forbid")

    fc_media: float = Field(..., ge=0, description="Frequência cardíaca média (bpm)")
    fc_desvio: float = Field(..., ge=0, description="Desvio padrão da frequência cardíaca (bpm)")
    hrv_rmssd: float = Field(..., ge=0, description="Variabilidade cardíaca, RMSSD (ms)")
    hrv_sdnn: float = Field(..., ge=0, description="Variabilidade cardíaca, SDNN (ms)")
    eda_media: float = Field(..., ge=0, description="Condutância média da pele (µS)")
    eda_picos: int = Field(
        ..., ge=0, description="Número de respostas de condutância na janela (picos de 0,01 µS ou mais)"
    )
    eda_inclinacao: float = Field(..., description="Tendência da condutância na janela (µS/min)")
    temp_media: float = Field(..., description="Temperatura média da pele (°C)")
    temp_inclinacao: float = Field(..., description="Tendência da temperatura na janela (°C/min)")
    movimento_media: float = Field(
        ..., ge=0, description="Aceleração média descontada a gravidade, |magnitude - 1 g| (g)"
    )


class JanelaFeatures(BaseModel):
    """Entrada do POST /prever: as features de uma janela de um usuário."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "usuario": "S2",
                    "fim_janela": "2026-11-27T14:04:00Z",
                    "duracao_s": 60,
                    "features": {
                        "fc_media": 92.0,
                        "fc_desvio": 6.3,
                        "hrv_rmssd": 24.0,
                        "hrv_sdnn": 38.5,
                        "eda_media": 0.52,
                        "eda_picos": 6,
                        "eda_inclinacao": 0.04,
                        "temp_media": 32.8,
                        "temp_inclinacao": -0.10,
                        "movimento_media": 0.03,
                    },
                }
            ]
        },
    )

    usuario: str = Field(..., min_length=1, description="Identificador do usuário (ex.: S2)")
    fim_janela: AwareDatetime = Field(
        ..., description="Fim da janela, com fuso horário (ex.: 2026-11-27T14:04:00Z)"
    )
    duracao_s: int = Field(60, gt=0, description="Duração da janela em segundos")
    features: Features


class Predicao(BaseModel):
    """Saída do POST /prever: estado atual, previsão e intervenção sugerida."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "usuario": "S2",
                    "fim_janela": "2026-11-27T14:04:00Z",
                    "estado": "estresse",
                    "probabilidade": 0.81,
                    "proxima_janela": "estresse",
                    "prob_proxima": 0.74,
                    "intervencao": "respiracao_guiada",
                    "versao_modelo": "falso-v0",
                }
            ]
        },
    )

    usuario: str = Field(..., description="Mesmo usuário da entrada")
    fim_janela: AwareDatetime = Field(..., description="Mesma janela da entrada")
    estado: Estado = Field(..., description="Estado estimado na janela atual")
    probabilidade: float = Field(..., ge=0, le=1, description="Confiança no estado atual")
    proxima_janela: Estado = Field(..., description="Estado previsto para a próxima janela")
    prob_proxima: float = Field(..., ge=0, le=1, description="Confiança na previsão")
    intervencao: Intervencao = Field(..., description="Intervenção sugerida")
    versao_modelo: str = Field(..., description="Versão do modelo que respondeu")
