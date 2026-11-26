FROM python:3.12-slim
WORKDIR /app
COPY ingestao/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY contratos/ contratos/
COPY shared/ shared/
COPY infra/banco/init/03-janelas-ia.sql schema.sql
COPY ingestao/janelas/processar.py .
CMD ["python", "-u", "processar.py"]
