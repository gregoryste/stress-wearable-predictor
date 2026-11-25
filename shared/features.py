"""Extração de features de uma janela de sinais do pulso (versão falsa v0).

Este módulo é usado em dois lugares e precisa ser o MESMO código nos dois:
- no treino do modelo (Gregory), com os sinais do WESAD;
- no serviço de ingestão (Jackson), com os sinais que chegam pelo MQTT.

Nesta versão, ``extrair_features`` valida a entrada e devolve valores fixos.
Quando a versão real ficar pronta, só o cálculo muda: os nomes das funções,
os parâmetros e o formato de saída (``contratos.api.Features``) continuam iguais.
"""

from collections.abc import Iterable, Mapping, Sequence

from contratos.api import Features

# Frequências de amostragem da pulseira Empatica E4 no WESAD (Hz).
FREQUENCIAS_HZ = {"bvp": 64, "eda": 4, "temp": 4, "acc": 32}

# Duração de uma janela, em segundos.
DURACAO_JANELA_S = 60

SINAIS_OBRIGATORIOS = tuple(FREQUENCIAS_HZ)


def amostras_esperadas(sinal: str, duracao_s: int = DURACAO_JANELA_S) -> int:
    """Quantidade de amostras de um sinal numa janela completa (ex.: bvp, 60 s -> 3840)."""
    if sinal not in FREQUENCIAS_HZ:
        raise ValueError(f"Sinal desconhecido: {sinal!r}")
    return FREQUENCIAS_HZ[sinal] * duracao_s


def juntar_mensagens(mensagens: Iterable[Mapping]) -> dict[str, list]:
    """Junta os sinais de várias mensagens MQTT (formato do simulador) numa só janela.

    As mensagens devem vir em ordem de chegada. O resultado tem uma lista por sinal,
    pronta para ``extrair_features``.

    Levanta ``ValueError`` se alguma mensagem estiver fora do formato do contrato,
    inclusive se faltar algum sinal: juntar mensagens incompletas desalinharia
    os sinais no tempo.
    """
    janela: dict[str, list] = {sinal: [] for sinal in SINAIS_OBRIGATORIOS}
    for posicao, mensagem in enumerate(mensagens):
        if not isinstance(mensagem, Mapping):
            raise ValueError(f"Mensagem {posicao} não é um objeto JSON")
        sinais = mensagem.get("sinais")
        if not isinstance(sinais, Mapping):
            raise ValueError(f"Mensagem {posicao} sem o campo 'sinais' no formato do contrato")
        for sinal in SINAIS_OBRIGATORIOS:
            valores = sinais.get(sinal)
            if valores is None:
                raise ValueError(f"Mensagem {posicao} sem o sinal obrigatório {sinal!r}")
            if not isinstance(valores, list):
                raise ValueError(f"Mensagem {posicao}: o sinal {sinal!r} precisa ser uma lista")
        for sinal in SINAIS_OBRIGATORIOS:
            janela[sinal].extend(sinais[sinal])
    return janela


def _validar_janela(sinais: Mapping[str, Sequence]) -> None:
    for sinal in SINAIS_OBRIGATORIOS:
        valores = sinais.get(sinal)
        if valores is None:
            raise ValueError(f"Sinal obrigatório ausente na janela: {sinal!r}")
        if len(valores) == 0:
            raise ValueError(f"Sinal vazio na janela: {sinal!r}")


def extrair_features(sinais: Mapping[str, Sequence]) -> Features:
    """Calcula as features de uma janela de 60 s.

    Parâmetros:
        sinais: uma sequência de valores por sinal, com as chaves
            ``bvp``, ``eda``, ``temp`` e ``acc`` (acc com triplas x, y, z).
            Numa janela completa, cada sinal tem ``amostras_esperadas(sinal)`` valores.

    Retorna:
        ``Features``, no formato do contrato da API.

    Versão falsa v0: valida a entrada e devolve valores fixos.
    """
    _validar_janela(sinais)
    return Features(
        fc_media=92.0,
        fc_desvio=6.3,
        hrv_rmssd=24.0,
        hrv_sdnn=38.5,
        eda_media=0.52,
        eda_picos=6,
        eda_inclinacao=0.04,
        temp_media=32.8,
        temp_inclinacao=-0.10,
        movimento_media=0.03,
    )
