"""API de IA: recebe as features de uma janela e responde estado, previsão e intervenção.

O modelo é carregado de modelos/modelo_estresse.joblib (ou do caminho em MODELO_PATH)
na primeira requisição. Se o arquivo não existir, /saude e /prever respondem erro,
e o teste de saúde do Docker marca o serviço como fora do ar.

Para rodar localmente:
    uvicorn api_ia.main:app --reload --port 8000
Documentação automática: http://localhost:8000/docs
"""

from functools import lru_cache

from fastapi import Depends, FastAPI

from api_ia.modelo import ServicoModelo
from contratos.api import JanelaFeatures, Predicao

app = FastAPI(
    title="API de IA - Stress Wearable Predictor",
    description="Recebe as features de uma janela de 60 s e responde o estado atual, "
    "a previsão da próxima janela e a intervenção sugerida.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def obter_servico() -> ServicoModelo:
    return ServicoModelo()


@app.get("/saude")
def saude(servico: ServicoModelo = Depends(obter_servico)) -> dict:
    """Indica que a API está no ar, qual modelo está carregado e quantos usuários têm calibração."""
    return {
        "status": "ok",
        "versao_modelo": servico.versao,
        "usuarios_calibrados": len(servico.usuarios_calibrados),
    }


@app.post("/prever", response_model=Predicao)
def prever(janela: JanelaFeatures, servico: ServicoModelo = Depends(obter_servico)) -> Predicao:
    """Classifica a janela, projeta a próxima e sugere uma intervenção."""
    return servico.prever(janela)
