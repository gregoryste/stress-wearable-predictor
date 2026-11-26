import os
from datetime import timezone

import pandas as pd
import psycopg
import streamlit as st

st.set_page_config(page_title="Painel do wearable")
st.title("Painel do wearable")

FONTE = os.getenv("DASHBOARD_FONTE", "exemplo").strip().lower()
if FONTE == "mqtt":
    SQL_PARTICIPANTES = "SELECT DISTINCT usuario FROM demo.mensagens_mqtt ORDER BY usuario"
    SQL_LEITURAS = """
        SELECT recebido_em, (payload -> 'sinais' -> 'temp' ->> -1)::double precision
        FROM demo.mensagens_mqtt
        WHERE usuario = %s
          AND jsonb_typeof(payload -> 'sinais' -> 'temp' -> -1) = 'number'
        ORDER BY recebido_em DESC, id DESC LIMIT 500
    """
    st.caption("Leituras MQTT · Atualização a cada 5 segundos")
elif FONTE == "exemplo":
    SQL_PARTICIPANTES = "SELECT DISTINCT participante FROM demo.sinais ORDER BY participante"
    SQL_LEITURAS = """
        SELECT instante, temperatura_c::double precision FROM demo.sinais
        WHERE participante = %s ORDER BY instante DESC LIMIT 500
    """
    st.caption("Dados de demonstração · TimescaleDB")
else:
    st.error("DASHBOARD_FONTE deve ser 'mqtt' ou 'exemplo'.")
    st.stop()


def horario(instante):
    return instante.astimezone(timezone.utc).strftime("%d/%m/%Y às %H:%M:%S UTC")


def exibir_predicao(predicao):
    st.subheader("Última predição")
    if predicao is None:
        st.info("Aguardando a primeira predição deste participante.")
        return

    fim, resposta = predicao
    st.caption(f"Fim da janela: {horario(fim)}")
    atual, proxima = st.columns(2)
    for coluna, titulo, estado, probabilidade in (
        (atual, "Estado atual", resposta["estado"], resposta["probabilidade"]),
        (proxima, "Próxima janela", resposta["proxima_janela"], resposta["prob_proxima"]),
    ):
        coluna.metric(titulo, estado.capitalize())
        percentual = f"{probabilidade:.2%}".replace(".", ",")
        coluna.caption(f"Probabilidade: {percentual}")
    st.write(f"Intervenção: {resposta['intervencao']}")


@st.fragment(run_every=5 if FONTE == "mqtt" else None)
def painel():
    st.button("Atualizar leituras")
    predicao = None
    try:
        with psycopg.connect(connect_timeout=5) as conn:
            participantes = [r[0] for r in conn.execute(SQL_PARTICIPANTES)]
            if not participantes:
                st.info("Aguardando a primeira leitura.")
                return
            participante = st.selectbox("Participante", participantes, key="participante")
            leituras = conn.execute(SQL_LEITURAS, (participante,)).fetchall()
            if FONTE == "mqtt" and conn.execute(
                "SELECT to_regclass('demo.predicoes') IS NOT NULL"
            ).fetchone()[0]:
                predicao = conn.execute(
                    """SELECT fim_janela, resposta FROM demo.predicoes
                       WHERE usuario = %s ORDER BY fim_janela DESC, ultimo_id DESC LIMIT 1""",
                    (participante,),
                ).fetchone()
    except psycopg.Error:
        st.error("Confira a conexão com o banco e as tabelas da fonte selecionada.")
        return

    if FONTE == "mqtt":
        exibir_predicao(predicao)
    if not leituras:
        st.info("Sem leituras de temperatura para este participante.")
        return

    instante, temperatura = leituras[0]
    st.metric("Última temperatura", f"{temperatura:.1f}".replace(".", ",") + " °C")
    st.caption(f"Última leitura: {horario(instante)}")
    historico = pd.DataFrame(leituras[::-1], columns=["Instante", "Temperatura (°C)"])
    st.subheader("Histórico de temperatura")
    st.line_chart(historico, x="Instante", y="Temperatura (°C)")


painel()
