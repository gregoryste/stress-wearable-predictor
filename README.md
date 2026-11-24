# Monitoramento de estresse com IoT e IA

Projeto da disciplina de IA/IoT da Unisinos. A infraestrutura inicial inclui um broker MQTT Mosquitto e um banco TimescaleDB com duas leituras fictícias de temperatura. O simulador, a ingestão, a IA e o dashboard são etapas futuras.

- [Proposta do projeto](./contexto/proposta.md)
- [Divisão de tarefas](./contexto/divisao-tarefas.md)

## Executar no Windows

Com o Docker Desktop em execução, abra no VS Code a pasta que contém este README. No terminal PowerShell integrado, crie a configuração local na primeira execução.

```powershell
Copy-Item .env.example .env
docker compose up -d --wait
docker compose ps
```

O exemplo inclui uma senha de demonstração para uso local. Se já executou o pacote inicial, o Compose reutiliza o volume do projeto iot-estresse e a senha original. Em um volume novo, o banco executa o SQL de infra/banco/init na primeira inicialização.

## Conferir o banco

```powershell
docker compose exec banco psql -U jackson -d iot -c "SELECT * FROM demo.sinais ORDER BY instante;"
```

O resultado esperado contém duas leituras do participante exemplo, com temperaturas de 32.1 e 32.2.

## Conferir o MQTT

No primeiro terminal, aguarde uma mensagem.

```powershell
docker compose exec mqtt mosquitto_sub -t teste/jackson -C 1 -v
```

Abra outro terminal na mesma pasta e publique a mensagem.

```powershell
docker compose exec mqtt mosquitto_pub -t teste/jackson -m "ola Jackson"
```

O primeiro terminal deve exibir `teste/jackson ola Jackson`. O broker atende o Windows em localhost:1883 e os serviços do Compose em mqtt:1883. O banco fica disponível na rede do Compose em banco:5432.

## Encerrar

```powershell
docker compose down
```

Esse comando preserva os dados no volume para a próxima execução.
