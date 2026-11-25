from datetime import timezone

import os
import pandas as pd
import psycopg
import streamlit as st

st.set_page_config(page_title="Temperatura do wearable")
st.title("Temperatura do wearable")
st.caption("Dados de demonstração · TimescaleDB")
st.button("Atualizar leituras")

try:
    with psycopg.connect(connect_timeout=5) as conn:
        fonte = os.getenv("DASHBOARD_FONTE", "exemplo").strip().lower()
        if fonte == "mqtt":
            sql_participantes = """
                SELECT DISTINCT usuario
                FROM demo.mensagens_mqtt
                ORDER BY usuario
            """
            sql_leituras = """
                SELECT recebido_em,
                    (payload -> 'sinais' -> 'temp' ->> -1)::double precision
                FROM demo.mensagens_mqtt
                WHERE usuario = %s
                AND jsonb_typeof(payload -> 'sinais' -> 'temp' -> -1) = 'number'
                ORDER BY recebido_em DESC, id DESC
                LIMIT 500
            """
        elif fonte == "exemplo":
            sql_participantes = """
                SELECT DISTINCT participante
                FROM demo.sinais
                ORDER BY participante
            """
            sql_leituras = """
                SELECT instante, temperatura_c::double precision
                FROM demo.sinais
                WHERE participante = %s
                ORDER BY instante DESC
                LIMIT 500
            """
        else:
            st.error("DASHBOARD_FONTE deve ser 'mqtt' ou 'exemplo'.")
            st.stop()

        participantes = conn.execute(sql_participantes).fetchall()

        if not participantes:
            st.info("Aguardando a primeira leitura.")
            st.stop()

        participante = st.selectbox("Participante", [row[0] for row in participantes])
        leituras = conn.execute(sql_leituras, (participante,)).fetchall() 
except psycopg.Error:
    st.error("Confira se o banco está acessível e se a tabela demo.sinais existe.")
    st.stop()

if not leituras:
    st.info("Sem leituras para este participante.")
    st.stop()

instante, temperatura = leituras[0]
st.metric("Última temperatura", f"{temperatura:.1f}".replace(".", ",") + " °C")
st.caption(instante.astimezone(timezone.utc).strftime("Última leitura em %d/%m/%Y às %H:%M:%S UTC"))
historico = pd.DataFrame(reversed(leituras), columns=["Instante", "Temperatura (°C)"])
st.subheader("Histórico de temperatura")
st.line_chart(historico, x="Instante", y="Temperatura (°C)")
