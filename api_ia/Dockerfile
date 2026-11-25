# Montado a partir da raiz do repositório (ver compose.api.yaml),
# para a imagem incluir o contrato em contratos/ e o modelo em modelos/.
FROM python:3.12-slim
WORKDIR /app
COPY api_ia/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY contratos/ contratos/
COPY modelos/ modelos/
COPY api_ia/ api_ia/
EXPOSE 8000
CMD ["uvicorn", "api_ia.main:app", "--host", "0.0.0.0", "--port", "8000"]
