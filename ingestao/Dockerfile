# Montado a partir da raiz do repositório (ver compose.ingestao.yaml),
# para a imagem incluir shared/ (extração de features) e contratos/.
FROM python:3.12-slim
WORKDIR /app
COPY ingestao/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY contratos/ contratos/
COPY ingestao/consumidor.py .
CMD ["python", "-u", "consumidor.py"]
