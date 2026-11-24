from simulador.simulador_mqtt import gerar_mensagem_exemplo


def test_contrato_mensagem_mqtt():
    msg = gerar_mensagem_exemplo("S2")

    # Campos obrigatórios na raiz da mensagem
    assert "usuario" in msg
    assert "timestamp" in msg
    assert "sinais" in msg
    assert "contexto" in msg

    # Sinais obrigatórios da pulseira (Empatica E4)
    for campo in ("bvp", "eda", "temp", "acc"):
        assert campo in msg["sinais"]
        assert isinstance(msg["sinais"][campo], list)
        assert len(msg["sinais"][campo]) > 0

    # Tipos básicos e contexto simulado
    assert isinstance(msg["usuario"], str)
    assert isinstance(msg["timestamp"], str)
    assert isinstance(msg["contexto"]["horario"], str)
    assert isinstance(msg["contexto"]["ruido_db"], (int, float))
