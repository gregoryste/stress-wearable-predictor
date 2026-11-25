"""Previsão por tendência e regras de intervenção.

No WESAD, nenhum par de minutos seguidos muda de estado (as condições duram de
6 a 20 minutos), então não há como treinar um modelo de "próximo minuto". A
previsão é feita por tendência: a API guarda as últimas janelas de cada usuário e
projeta a probabilidade de estresse da próxima janela.
"""

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from contratos.api import Intervencao

LIMIAR_ESTRESSE = 0.5
# Quantas janelas recentes entram no cálculo da tendência.
JANELAS_TENDENCIA = 3
# Estresse por tantas janelas seguidas (cerca de 10 minutos) pede uma pausa maior.
JANELAS_ESTRESSE_LONGO = 10
# Um intervalo maior que este entre duas janelas começa uma nova sessão.
INTERVALO_NOVA_SESSAO = timedelta(minutes=5)
HISTORICO_MAXIMO = 20


@dataclass(frozen=True)
class Registro:
    """Resultado de uma janela, guardado para calcular tendência e duração do estresse."""

    fim_janela: datetime
    estado: str
    prob_estresse: float


class Historico:
    """Últimas janelas de cada usuário, em memória. Seguro para várias requisições ao mesmo tempo."""

    def __init__(self, maximo: int = HISTORICO_MAXIMO):
        self._por_usuario: dict[str, deque[Registro]] = {}
        self._maximo = maximo
        self._trava = threading.Lock()

    def registrar(self, usuario: str, registro: Registro) -> list[Registro]:
        """Acrescenta a janela e devolve o histórico atual do usuário (mais antiga primeiro)."""
        with self._trava:
            fila = self._por_usuario.setdefault(usuario, deque(maxlen=self._maximo))
            if fila:
                ultimo = fila[-1]
                if registro.fim_janela == ultimo.fim_janela:
                    fila.pop()  # mesma janela reenviada: substitui em vez de duplicar
                elif (
                    registro.fim_janela < ultimo.fim_janela
                    or registro.fim_janela - ultimo.fim_janela > INTERVALO_NOVA_SESSAO
                ):
                    fila.clear()  # janela fora de ordem ou após uma pausa longa: nova sessão
            fila.append(registro)
            return list(fila)


def projetar_estresse(registros: list[Registro]) -> float:
    """Probabilidade de estresse projetada para a próxima janela (tendência linear)."""
    recentes = [r.prob_estresse for r in registros[-JANELAS_TENDENCIA:]]
    if len(recentes) < 2:
        return recentes[-1]
    inclinacao = (recentes[-1] - recentes[0]) / (len(recentes) - 1)
    return min(1.0, max(0.0, recentes[-1] + inclinacao))


def prever_proxima(registros: list[Registro], probabilidades: dict[str, float]) -> tuple[str, float]:
    """Estado previsto para a próxima janela e a confiança nessa previsão."""
    projetada = projetar_estresse(registros)
    if projetada >= LIMIAR_ESTRESSE:
        return "estresse", projetada
    sem_estresse = {estado: p for estado, p in probabilidades.items() if estado != "estresse"}
    return max(sem_estresse, key=sem_estresse.get), 1.0 - projetada


def escolher_intervencao(registros: list[Registro], proxima_janela: str) -> Intervencao:
    """Intervenção sugerida, da mais leve à mais forte.

    - estresse agora, por pouco tempo -> respiração guiada
    - estresse por ``JANELAS_ESTRESSE_LONGO`` janelas seguidas -> pausa ativa
    - sem estresse agora, mas tendência de estresse -> aviso antecipado
    - caso contrário -> nenhuma

    ``ajuste_ambiente`` depende do contexto (ruído, temperatura da sala) e
    ``encaminhamento`` de um padrão de vários dias; nenhum dos dois chega à API
    por janela, então ficam para quem tem esses dados (dashboard).
    """
    if registros[-1].estado == "estresse":
        seguidas = 0
        for registro in reversed(registros):
            if registro.estado != "estresse":
                break
            seguidas += 1
        if seguidas >= JANELAS_ESTRESSE_LONGO:
            return Intervencao.pausa_ativa
        return Intervencao.respiracao_guiada
    if proxima_janela == "estresse":
        return Intervencao.aviso_antecipado
    return Intervencao.nenhuma
