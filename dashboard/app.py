import os
from datetime import timezone

import altair as alt
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

# Últimas janelas de 60 s exibidas no histórico (2 horas em velocidade normal).
JANELAS_HISTORICO = 120
SQL_PREDICOES = """
    SELECT fim_janela, resposta FROM demo.predicoes
    WHERE usuario = %s ORDER BY fim_janela DESC, ultimo_id DESC LIMIT %s
"""

ESTADOS = {"repouso": "Repouso", "estresse": "Estresse", "diversao": "Diversão"}
CORES_ESTADOS = {"Repouso": "#2a78d6", "Estresse": "#eb6834", "Diversão": "#3a9d5d"}

# Valores de ``intervencao`` no contrato (contratos/api.py) -> (tipo da caixa, título, detalhe).
# "nenhuma" não aparece aqui: sem intervenção, não há alerta.
INTERVENCOES = {
    "aviso_antecipado": (
        "warning", "Sinais de estresse chegando",
        "A tendência dos últimos minutos indica estresse no próximo minuto.",
    ),
    "respiracao_guiada": (
        "error", "Estresse detectado: que tal 1 minuto de respiração guiada?",
        "O estresse se manteve por 2 minutos seguidos.",
    ),
    "pausa_ativa": (
        "error", "Estresse há cerca de 10 minutos: faça uma pausa",
        "Levante, caminhe um pouco ou beba água antes de voltar à atividade.",
    ),
    "ajuste_ambiente": (
        "warning", "Ajuste o ambiente",
        "Ruído ou temperatura do ambiente podem estar contribuindo para o estresse.",
    ),
    "encaminhamento": (
        "warning", "Estresse frequente nos últimos dias",
        "Vale conversar com um profissional de saúde.",
    ),
}
ICONES = {"warning": "⚠️", "error": "🚨"}


def horario(instante):
    return instante.astimezone(timezone.utc).strftime("%d/%m/%Y às %H:%M:%S UTC")


def percentual(valor):
    return f"{valor:.0%}".replace(".", ",")


def chance_estresse_proxima(resposta):
    """Chance de estresse projetada para a próxima janela.

    A API devolve a confiança no estado previsto: se a previsão é "estresse",
    ``prob_proxima`` já é a chance de estresse; se não, é 1 menos essa chance.
    """
    if resposta["proxima_janela"] == "estresse":
        return resposta["prob_proxima"]
    return 1.0 - resposta["prob_proxima"]


def exibir_alerta(resposta):
    """O alerta segue a intervenção, não o estado: a API só sugere uma intervenção
    de estresse depois de 2 janelas seguidas, o que corta alarmes por picos isolados."""
    intervencao = INTERVENCOES.get(resposta["intervencao"])
    if intervencao:
        tipo, titulo, detalhe = intervencao
        getattr(st, tipo)(f"**{titulo}**  \n{detalhe}", icon=ICONES[tipo])
    elif resposta["estado"] == "estresse":
        st.info("Possível estresse neste minuto. O alerta só aparece se continuar "
                "por 2 minutos seguidos.", icon="⏳")
    else:
        st.success("Sem sinais de estresse.", icon="✅")


def exibir_predicao(predicoes):
    st.subheader("Última predição")
    if not predicoes:
        st.info("Aguardando a primeira predição deste participante.")
        return

    fim, resposta = predicoes[0]
    st.caption(f"Fim da janela: {horario(fim)}")
    exibir_alerta(resposta)
    atual, proxima = st.columns(2)
    for coluna, titulo, estado, probabilidade in (
        (atual, "Estado atual", resposta["estado"], resposta["probabilidade"]),
        (proxima, "Próxima janela", resposta["proxima_janela"], resposta["prob_proxima"]),
    ):
        coluna.metric(titulo, ESTADOS.get(estado, estado.capitalize()))
        coluna.caption(f"Confiança: {percentual(probabilidade)}")


def exibir_historico(predicoes):
    if len(predicoes) < 2:
        return
    historico = pd.DataFrame([
        {
            "Instante": fim,
            "Chance de estresse (%)": round(100 * chance_estresse_proxima(resposta), 1),
            "Estado": ESTADOS.get(resposta["estado"], resposta["estado"]),
            "Alerta": INTERVENCOES[resposta["intervencao"]][1]
            if resposta["intervencao"] in INTERVENCOES else "",
        }
        for fim, resposta in reversed(predicoes)
    ])
    eixo_x = alt.X("Instante:T", title=None, axis=alt.Axis(format="%H:%M:%S", labelOverlap=True))
    eixo_y = alt.Y("Chance de estresse (%):Q", scale=alt.Scale(domain=[0, 100]))
    linha = alt.Chart(historico).mark_line(color="#888888", strokeWidth=1.5).encode(eixo_x, eixo_y)
    pontos = alt.Chart(historico).mark_circle(size=60, opacity=1).encode(
        eixo_x, eixo_y,
        color=alt.Color(
            "Estado:N", title="Estado da janela",
            scale=alt.Scale(domain=list(CORES_ESTADOS), range=list(CORES_ESTADOS.values())),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[alt.Tooltip("Instante:T", format="%H:%M:%S"), "Estado:N",
                 "Chance de estresse (%):Q", "Alerta:N"],
    )
    limiar = alt.Chart().mark_rule(strokeDash=[4, 4], color="#999999").encode(y=alt.datum(50))
    camadas = [linha, limiar, pontos]
    alertas = historico[historico["Alerta"] != ""]
    if not alertas.empty:
        # Faixa vermelha ao fundo em cada janela com alerta.
        camadas.insert(0, alt.Chart(alertas).mark_rule(color="#e74c3c", opacity=0.2, strokeWidth=8).encode(
            eixo_x, tooltip=[alt.Tooltip("Instante:T", format="%H:%M:%S"), "Alerta:N"],
        ))

    st.subheader("Histórico de estresse")
    st.altair_chart(alt.layer(*camadas), use_container_width=True)
    st.caption(f"Últimas {len(historico)} janelas de 60 s. A linha é a chance de estresse projetada "
               "para o minuto seguinte; cada ponto tem a cor do estado da janela; as faixas vermelhas "
               "marcam os alertas.")


@st.fragment(run_every=5 if FONTE == "mqtt" else None)
def painel():
    st.button("Atualizar leituras")
    predicoes = []
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
                predicoes = conn.execute(SQL_PREDICOES, (participante, JANELAS_HISTORICO)).fetchall()
    except psycopg.Error:
        st.error("Confira a conexão com o banco e as tabelas da fonte selecionada.")
        return

    if FONTE == "mqtt":
        exibir_predicao(predicoes)
        exibir_historico(predicoes)
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
