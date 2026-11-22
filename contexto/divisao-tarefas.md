# Divisão de tarefas

Projeto final IoT + IA: monitoramento de estresse sensível ao contexto. Entrega: **30/11/2026**.
Detalhes da solução em [proposta.md](proposta.md).

O trabalho está dividido em três frentes que andam em paralelo:

- **Gregory:** IA e contexto, a parte mais incerta do projeto.
- **Lucas:** relatório, documentação e qualidade, além do simulador do wearable.
- **Jackson:** infraestrutura e interface, o ponto onde todas as peças se conectam.

A carga foi equilibrada para que o Jackson não concentre peças demais: o publicador de exemplo, os testes de contrato e o README ficam com o Lucas.

## Como ninguém fica esperando ninguém

A regra é **definir as interfaces no primeiro dia e trabalhar com versões falsas**. Cada pessoa começa contra uma imitação da peça do colega, que responde no formato combinado com dados inventados. Quando a peça real fica pronta, ela entra no lugar da falsa sem mudar o formato, e ninguém para.

| Quem esperaria | Esperaria por | Como não esperar | Quem entrega a versão falsa |
| --- | --- | --- | --- |
| Ingestão (Jackson) | Sinais do simulador | A primeira versão do simulador publica um JSON de exemplo no broker em loop | Lucas, em 22/11 |
| Ingestão (Jackson) | Módulo de features | Função com o nome e os campos finais que, por enquanto, devolve valores fixos | Gregory, em 22/11 |
| Ingestão (Jackson) | API de IA | API que já roda mas sempre devolve a mesma resposta de exemplo | Gregory, em 22/11 |
| Dashboard (Jackson) | Dados reais no banco | Script que preenche o banco com dados de exemplo | Jackson, em 22/11 |
| Simulador (Lucas) | Broker | Mosquitto em Docker sobe em minutos; dá para testar sozinho | Lucas |
| API (Gregory) | Modelo treinado | Começa com a resposta fixa e troca pelo modelo quando ele estiver pronto | Gregory |
| Relatório (Lucas) | Métricas e prints | Escreve antes as seções que não dependem de resultado e deixa tabelas e figuras marcadas como "[preencher]"; o script de treino gera `metricas.json` e gráficos automaticamente | Gregory (arquivo de métricas) |

### O que não dá para eliminar, só encurtar

1. **A reunião dos contratos:** precisa dos três juntos, mas é uma vez só e curta (22/11).
2. **Os números finais do relatório:** o texto fica pronto antes; quando o treino gerar `metricas.json` e os gráficos, o Lucas só atualiza tabelas e figuras.
3. **A gravação do vídeo:** precisa do sistema funcionando, por isso fica em 29/11. O roteiro é escrito antes.

## Contratos (22/11, os três juntos, 1 a 2 horas)

Os contratos ficam **como arquivos no repositório**, não só combinados de boca:

| Arquivo | Conteúdo | Usado por |
| --- | --- | --- |
| `contratos/mensagem_mqtt.json` | Tópico (ex.: `wearable/{usuario}/sinais`) e exemplo da mensagem: timestamp, sinal, valores | Lucas (publica), Jackson (consome) |
| `contratos/schema.sql` | Tabelas: sinais brutos, features por janela, predições, retorno das intervenções | Jackson (ingestão e dashboard), Lucas (relatório) |
| `contratos/api.py` | Formato de entrada (features de uma janela) e de saída (estado, probabilidade, previsão da próxima janela, intervenção); o FastAPI gera a documentação automaticamente | Gregory (API), Jackson (ingestão) |
| `shared/features.py` | Função de extração de features com nome e campos finais, inicialmente devolvendo valores fixos | Gregory (treino), Jackson (ingestão) |
| `exemplos/` | Uma janela de sinais e uma resposta da API de exemplo, para testes | Todos |

> **Atenção:** a extração de features deve ser **o mesmo código** no treino e no fluxo ao vivo. Se não for, o modelo recebe dados diferentes dos que aprendeu. O módulo fica com o Gregory, e o serviço de ingestão do Jackson apenas o importa.

## Regras do dia a dia

- Cada um trabalha no seu branch e abre pull request para o principal.
- Os testes automáticos (GitHub Actions, mantidos pelo Lucas) conferem se cada peça respeita os contratos. Se alguém quebrar o formato, o teste avisa antes de atrapalhar os outros.
- Mudar um contrato só com o acordo dos três.
- **Até 23/11:** todas as peças falsas ligadas e o fluxo completo funcionando de ponta a ponta, mesmo sem dados reais. A partir daí, cada um só troca a sua peça falsa pela real.

## Responsáveis

| | Gregory: IA e contexto | Lucas: relatório, qualidade e simulador | Jackson: infraestrutura e interface |
| --- | --- | --- | --- |
| **Desenvolve** | Módulo de extração de features, modelos de estado e previsão, validação leave-one-subject-out, padrão pessoal de cada usuário, regras de intervenção, API de IA (FastAPI) | Simulador do wearable (reproduz o WESAD e publica no MQTT), contexto simulado (horário, ruído), testes de contrato no GitHub Actions | Docker Compose, broker Mosquitto, banco TimescaleDB, serviço de ingestão, dashboard (Streamlit), alertas e Telegram, botão "isso ajudou?" |
| **Versões falsas (22/11)** | `shared/features.py` e API com resposta fixa | Primeira versão do simulador publicando a mensagem de exemplo em loop | Banco com dados de exemplo |
| **Documentação** | Script de treino gera `metricas.json` e gráficos; um parágrafo sobre os modelos para o Lucas (até 27/11) | **Relatório PDF completo:** escreve todas as seções, faz o diagrama de arquitetura (C4) e as decisões de arquitetura (ADRs), consolida e revisa. **README:** escreve e valida instalando do zero num clone novo | Comandos de instalação para o README; prints das telas e detalhes da infraestrutura para o Lucas (até 27/11) |
| **Vídeo** | Roteiro (escrito antes) e narração do pitch | Edição | Gravação da demo |

## Cronograma

| Data | Gregory | Lucas | Jackson |
| --- | --- | --- | --- |
| 22/11 | **Contratos (os três)**; `features.py` e API com resposta fixa; baixar o WESAD | **Contratos (os três)**; criar repositório; simulador publicando a mensagem de exemplo | **Contratos (os três)**; Docker Compose com broker e banco; dados de exemplo no banco |
| 23/11 | **Fluxo falso de ponta a ponta funcionando (os três)** | | |
| 23–25/11 | Features reais + modelo v1 com métricas | Simulador real publicando o WESAD (até 24/11); testes de contrato no GitHub Actions; relatório: contextualização, metodologia e arquitetura, a partir da proposta | Ingestão real, dashboard lendo do banco |
| **25/11** | | **Infraestrutura congelada** | |
| 26–27/11 | Modelo de previsão, padrão pessoal, regras de intervenção, modelo real na API; roteiro do vídeo; **até 27/11:** `metricas.json`, gráficos e texto para o Lucas | Relatório: desenvolvimento e funcionalidades (com prints dos dados de exemplo); contexto simulado no simulador; primeira versão do README | Alertas, Telegram, botão de retorno; **até 27/11:** comandos de instalação, prints e detalhes para o Lucas |
| 28/11 | Revisa as seções de IA do relatório | Relatório: preenche resultados, conclusões, revisão e PDF final; valida o README num clone novo | Corrige o que a validação do README apontar; ajustes finais |
| 29/11 | Narração | Edição | Gravação da demo |
| 30/11 | Revisão final e entrega (os três) | | |

## Pontos de controle

Conversas rápidas de 15 minutos:

- **23/11:** fluxo falso funcionando de ponta a ponta.
- **25/11:** peças reais entrando no lugar das falsas, infraestrutura congelada, testes de contrato rodando.
- **27/11:** modelo real integrado, alertas chegando no painel, material do relatório entregue ao Lucas.

## Checklist

### Gregory
- [ ] Contratos definidos com o grupo
- [ ] `shared/features.py` com valores fixos (22/11)
- [ ] API de IA com resposta fixa (22/11)
- [ ] WESAD baixado e preparado
- [ ] Features reais no `shared/features.py`
- [ ] Modelo de estado com métricas leave-one-subject-out
- [ ] Modelo de previsão da próxima janela
- [ ] Padrão pessoal e regras de intervenção
- [ ] Modelo real servido pela API
- [ ] `metricas.json`, gráficos e texto sobre os modelos entregues ao Lucas (27/11)
- [ ] Roteiro e narração do vídeo

### Lucas
- [ ] Contratos definidos com o grupo
- [ ] Repositório público criado
- [ ] Simulador publicando a mensagem de exemplo em loop (22/11)
- [ ] Simulador publicando os sinais do WESAD no MQTT
- [ ] Contexto simulado (horário, ruído)
- [ ] Testes de contrato no GitHub Actions
- [ ] Relatório: contextualização, metodologia e arquitetura
- [ ] Diagrama de arquitetura (C4) e decisões de arquitetura (ADRs)
- [ ] Relatório: desenvolvimento e funcionalidades
- [ ] README escrito e validado num clone novo
- [ ] Relatório: resultados preenchidos e conclusões
- [ ] PDF final revisado
- [ ] Edição do vídeo

### Jackson
- [ ] Contratos definidos com o grupo
- [ ] Docker Compose com broker e banco (22/11)
- [ ] Dados de exemplo no banco (22/11)
- [ ] Fluxo falso de ponta a ponta (23/11, com o grupo)
- [ ] Serviço de ingestão gravando sinais, features e predições no banco
- [ ] Dashboard com sinais, estado e histórico
- [ ] Alertas, Telegram e botão "isso ajudou?"
- [ ] Comandos de instalação, prints e detalhes da infraestrutura entregues ao Lucas (27/11)
- [ ] Correções apontadas pela validação do README
- [ ] Gravação da demo
