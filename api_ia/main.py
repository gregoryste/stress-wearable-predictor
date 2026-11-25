"""API de IA (versão falsa v0).

Responde sempre a mesma predição, no formato do contrato, para que a ingestão
possa ser desenvolvida e testada antes de o modelo real ficar pronto.
Quando o modelo existir, só o corpo de ``prever`` muda; o contrato fica igual.

Para rodar localmente:
    uvicorn api_ia.main:app --reload --port 8000
Documentação automática: http://localhost:8000/docs
"""

from fastapi import FastAPI

from contratos.api import Estado, Intervencao, JanelaFeatures, Predicao

VERSAO_MODELO = "falso-v0"

app = FastAPI(
    title="API de IA - Stress Wearable Predictor",
    description="Recebe as features de uma janela de 60 s e responde o estado atual, "
    "a previsão da próxima janela e a intervenção sugerida.",
    version="0.1.0",
)


@app.get("/saude")
def saude() -> dict:
    """Indica que a API está no ar e qual modelo está carregado."""
    return {"status": "ok", "versao_modelo": VERSAO_MODELO}


@app.post("/prever", response_model=Predicao)
def prever(janela: JanelaFeatures) -> Predicao:
    """Versão falsa: resposta fixa, repetindo usuário e janela da entrada."""
    return Predicao(
        usuario=janela.usuario,
        fim_janela=janela.fim_janela,
        estado=Estado.estresse,
        probabilidade=0.81,
        proxima_janela=Estado.estresse,
        prob_proxima=0.74,
        intervencao=Intervencao.respiracao_guiada,
        versao_modelo=VERSAO_MODELO,
    )
