from datetime import timezone

import pandas as pd
import psycopg
import streamlit as st

st.set_page_config(page_title="Temperatura do wearable")
st.title("Temperatura do wearable")
st.caption("Dados de demonstração · TimescaleDB")
st.button("Atualizar leituras")

try:
    with psycopg.connect(connect_timeout=5) as conn:
        participantes = conn.execute(
            "SELECT DISTINCT participante FROM demo.sinais ORDER BY participante"
        ).fetchall()
        if not participantes:
            st.info("Aguardando a primeira leitura.")
            st.stop()
        participante = st.selectbox("Participante", [row[0] for row in participantes])
        leituras = conn.execute(
            """SELECT instante, temperatura_c::double precision FROM demo.sinais
               WHERE participante = %s ORDER BY instante DESC LIMIT 500""",
            (participante,),
        ).fetchall()
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
