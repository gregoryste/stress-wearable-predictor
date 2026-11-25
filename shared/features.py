"""Extração de features de uma janela de sinais do pulso (Empatica E4, WESAD).

Este módulo é usado em dois lugares e precisa ser o MESMO código nos dois:
- no treino do modelo (Gregory), com os sinais do WESAD;
- no serviço de ingestão (Jackson), com os sinais que chegam pelo MQTT.

Só depende de numpy e scipy, para a imagem da ingestão continuar leve.
A saída segue o contrato da API (``contratos.api.Features``).
"""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from contratos.api import Features

# Frequências de amostragem da pulseira Empatica E4 no WESAD (Hz).
FREQUENCIAS_HZ = {"bvp": 64, "eda": 4, "temp": 4, "acc": 32}

# Duração de uma janela, em segundos.
DURACAO_JANELA_S = 60

SINAIS_OBRIGATORIOS = tuple(FREQUENCIAS_HZ)

# Uma janela precisa ter pelo menos esta fração das amostras esperadas.
FRACAO_MINIMA_AMOSTRAS = 0.5

# O acelerômetro do E4 mede em unidades de 1/64 g (readme do WESAD).
ACC_UNIDADES_POR_G = 64.0

# Batimentos: faixa de frequência cardíaca considerada (40 a 200 bpm).
FC_MIN_HZ, FC_MAX_HZ = 0.7, 3.5
IBI_MIN_MS, IBI_MAX_MS = 300.0, 1500.0
# Descarta intervalos entre batimentos muito diferentes da mediana (artefatos).
TOLERANCIA_IBI = 0.25
# Mínimo de intervalos válidos para calcular frequência e variabilidade.
INTERVALOS_MINIMOS = 10

# Condutância da pele: menor pico considerado uma resposta (µS).
EDA_PROEMINENCIA_MIN_US = 0.01


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


def _como_array(sinais: Mapping[str, Sequence], sinal: str) -> np.ndarray:
    """Converte um sinal para array numpy e valida tamanho e valores."""
    valores = sinais.get(sinal)
    if valores is None:
        raise ValueError(f"Sinal obrigatório ausente na janela: {sinal!r}")
    if len(valores) == 0:
        raise ValueError(f"Sinal vazio na janela: {sinal!r}")
    try:
        array = np.asarray(valores, dtype=float)
    except (TypeError, ValueError) as erro:
        raise ValueError(f"Sinal {sinal!r} com valores não numéricos") from erro

    if sinal == "acc":
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("Sinal 'acc' precisa ser uma lista de triplas [x, y, z]")
    else:
        array = array.reshape(-1)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"Sinal {sinal!r} contém valores inválidos (NaN ou infinito)")

    minimo = int(FRACAO_MINIMA_AMOSTRAS * amostras_esperadas(sinal))
    if len(array) < minimo:
        raise ValueError(
            f"Janela incompleta: {sinal!r} tem {len(array)} amostras, "
            f"o mínimo é {minimo} ({FRACAO_MINIMA_AMOSTRAS:.0%} de {DURACAO_JANELA_S} s)"
        )
    return array


def _inclinacao_por_minuto(valores: np.ndarray, frequencia_hz: float) -> float:
    """Tendência linear do sinal, em unidades por minuto."""
    tempo_min = np.arange(len(valores)) / frequencia_hz / 60.0
    return float(np.polyfit(tempo_min, valores, 1)[0])


def _refinar_picos(sinal: np.ndarray, picos: np.ndarray) -> np.ndarray:
    """Posição de cada pico com precisão abaixo de uma amostra (interpolação parabólica).

    A 64 Hz, cada amostra vale 15,6 ms; sem o refinamento, a variabilidade cardíaca
    ficaria dominada por esse arredondamento.
    """
    internos = picos[(picos > 0) & (picos < len(sinal) - 1)]
    esquerda, centro, direita = sinal[internos - 1], sinal[internos], sinal[internos + 1]
    denominador = esquerda - 2 * centro + direita
    with np.errstate(divide="ignore", invalid="ignore"):
        deslocamento = np.where(denominador != 0, 0.5 * (esquerda - direita) / denominador, 0.0)
    return internos + np.clip(deslocamento, -0.5, 0.5)


def _intervalos_entre_batimentos(bvp: np.ndarray) -> np.ndarray:
    """Intervalos entre batimentos (ms) detectados no sinal de pulso (BVP)."""
    frequencia = FREQUENCIAS_HZ["bvp"]
    filtro = butter(3, [FC_MIN_HZ, FC_MAX_HZ], btype="bandpass", fs=frequencia, output="sos")
    filtrado = sosfiltfilt(filtro, bvp - np.mean(bvp))

    picos, _ = find_peaks(
        filtrado,
        distance=max(1, int(frequencia / FC_MAX_HZ)),
        prominence=0.3 * np.std(filtrado),
    )
    if len(picos) < 2:
        return np.empty(0)

    instantes_ms = _refinar_picos(filtrado, picos) / frequencia * 1000.0
    intervalos = np.diff(instantes_ms)
    intervalos = intervalos[(intervalos >= IBI_MIN_MS) & (intervalos <= IBI_MAX_MS)]
    if intervalos.size:
        mediana = np.median(intervalos)
        intervalos = intervalos[np.abs(intervalos - mediana) <= TOLERANCIA_IBI * mediana]
    return intervalos


def _picos_eda(eda: np.ndarray) -> int:
    """Número de respostas de condutância da pele na janela."""
    frequencia = FREQUENCIAS_HZ["eda"]
    # Média móvel de 1 s. As bordas são completadas com o próprio valor do sinal:
    # completar com zero criaria uma queda falsa no fim da janela (e um pico falso).
    estendido = np.pad(eda, (frequencia // 2, frequencia - 1 - frequencia // 2), mode="edge")
    suavizado = np.convolve(estendido, np.ones(frequencia) / frequencia, mode="valid")
    picos, _ = find_peaks(
        suavizado, prominence=EDA_PROEMINENCIA_MIN_US, distance=frequencia
    )
    return int(len(picos))


def extrair_features(sinais: Mapping[str, Sequence]) -> Features:
    """Calcula as features de uma janela de sinais do pulso.

    Parâmetros:
        sinais: uma sequência de valores por sinal, com as chaves
            ``bvp`` (64 Hz), ``eda`` (4 Hz, µS), ``temp`` (4 Hz, °C) e
            ``acc`` (32 Hz, triplas x, y, z em unidades de 1/64 g).
            Numa janela completa, cada sinal tem ``amostras_esperadas(sinal)`` valores;
            é aceito a partir de 50% disso.

    Retorna:
        ``Features``, no formato do contrato da API.

    Levanta ``ValueError`` se a janela estiver incompleta, com valores inválidos
    ou sem batimentos detectáveis. Quem chama (ingestão ou treino) deve
    descartar essa janela.
    """
    bvp = _como_array(sinais, "bvp")
    eda = _como_array(sinais, "eda")
    temp = _como_array(sinais, "temp")
    acc = _como_array(sinais, "acc")

    intervalos = _intervalos_entre_batimentos(bvp)
    if len(intervalos) < INTERVALOS_MINIMOS:
        raise ValueError(
            f"Batimentos insuficientes na janela: {len(intervalos)} intervalos válidos, "
            f"o mínimo é {INTERVALOS_MINIMOS}"
        )
    frequencia_instantanea = 60000.0 / intervalos

    magnitude_g = np.linalg.norm(acc, axis=1) / ACC_UNIDADES_POR_G

    return Features(
        fc_media=float(np.mean(frequencia_instantanea)),
        fc_desvio=float(np.std(frequencia_instantanea)),
        hrv_rmssd=float(np.sqrt(np.mean(np.diff(intervalos) ** 2))),
        hrv_sdnn=float(np.std(intervalos, ddof=1)),
        eda_media=float(max(np.mean(eda), 0.0)),
        eda_picos=_picos_eda(eda),
        eda_inclinacao=_inclinacao_por_minuto(eda, FREQUENCIAS_HZ["eda"]),
        temp_media=float(np.mean(temp)),
        temp_inclinacao=_inclinacao_por_minuto(temp, FREQUENCIAS_HZ["temp"]),
        movimento_media=float(np.mean(np.abs(magnitude_g - 1.0))),
    )
