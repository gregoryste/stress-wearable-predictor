# Proposta: Monitoramento de estresse sensível ao contexto com IoT e IA

Projeto final da disciplina de IA/IoT (Unisinos). Entrega: **30/11/2026**.

## Resumo

Propomos um **sistema sensível ao contexto que detecta e antecipa episódios de estresse** a partir de sinais de wearables. Sensores fisiológicos (IoT) alimentam um histórico de contexto (Big Data), sobre o qual um modelo de IA classifica o estado atual e prevê o próximo, disparando alertas e intervenções. A demo usa dados reais do dataset público **WESAD**, transmitidos por MQTT como se fossem um wearable ao vivo.

Entregáveis exigidos pelo enunciado:

- Relatório técnico em PDF
- Repositório público no GitHub com README claro
- Vídeo pitch e demo de até 5 minutos

---

## 1. Por que esta linha

O tema foi escolhido para servir também como base para o Mestrado em Computação Aplicada (Unisinos), com foco em **Engenharia de Software** e **Ambientes Inteligentes**.

- **Ambientes Inteligentes (Prof. Jorge Barbosa):** a disciplina trabalha sensibilidade ao contexto, históricos de contexto, predição de contexto e fenótipos digitais. A bibliografia é fortemente voltada à saúde (iAware para estresse e ansiedade, cuidado ubíquo de doenças crônicas, fenotipagem digital com aprendizado de máquina).
- **Engenharia de Software:** arquitetura, qualidade, testes e interoperabilidade entram na forma como o sistema é construído.
- **Disciplinas relacionadas:** Applied Computing in Healthcare (IoT na saúde, wearables, ética e LGPD), Conectividade Avançada e Sistemas Distribuídos (IoT, wearables, saúde) e Edge, Fog e Cloud.

A tríade do enunciado se encaixa diretamente nos conceitos da área:

| Enunciado | Conceito de Ambientes Inteligentes |
| --- | --- |
| IoT (sensores, wearables) | Captura de contexto |
| Big Data (séries históricas) | Histórico de contexto |
| IA (previsão, alertas) | Predição de contexto e fenótipo digital |

Benefícios para o mestrado:

- O processo seletivo de mestrado do PPGCA avalia por análise documental, incluindo produção técnico-científica e motivação. Um repositório público bem feito conta a favor. A 2ª janela de inscrições para 2027/1 vai de 14/12/2026 a 01/03/2027.
- O projeto pode virar um artigo curto (por exemplo, no SBCAS, Simpósio Brasileiro de Computação Aplicada à Saúde).
- O trabalho individual de Ambientes Inteligentes é ligado ao tema da dissertação; uma revisão sobre fenotipagem digital com IoT e wearables seria uma continuação natural.

---

## 2. Problema e justificativa

O estresse crônico costuma ser percebido tarde, quando já afeta sono, desempenho e saúde. Avaliações por questionário são pontuais e subjetivas; wearables, por outro lado, medem sinais fisiológicos ligados ao estresse de forma contínua e passiva.

- **Oportunidade de mercado:** smartwatches e pulseiras já são comuns, mas raramente transformam os sinais em ação preventiva personalizada.
- **Relevância acadêmica:** alinhado às linhas de Ambientes Inteligentes e Applied Computing in Healthcare.
- **Escopo deliberado:** o foco é estresse e bem-estar, não diagnóstico clínico, o que mantém o projeto viável e eticamente seguro no prazo.

---

## 3. Solução proposta

O sistema acompanha o usuário em tempo real, guarda a história do seu contexto e usa essa história para prever e agir antes do pico de estresse.

| Pilar | O que fazemos |
| --- | --- |
| IoT | Wearable (simulado com dados reais) publica sinais via MQTT a cada janela de tempo |
| Big Data | Séries temporais de todos os participantes e sessões em banco de séries temporais |
| IA | Classificação do estado atual, previsão do próximo estado e perfil individual de resposta ao estresse |
| Ação | Alerta no dashboard (e opcionalmente Telegram) com sugestão de intervenção |

Funcionalidades principais:

1. Painel ao vivo com sinais e estado estimado (repouso, estresse, diversão).
2. Alerta preditivo quando a probabilidade de estresse na próxima janela passa de um limiar.
3. Linha do tempo do histórico de contexto por usuário, com episódios marcados.
4. Relatório de padrões: horários e sinais que mais antecedem episódios (o fenótipo digital).

---

## 4. Arquitetura e tecnologias

Modelo borda, fog e nuvem: o wearable só publica, o gateway prepara os dados e a nuvem guarda, aprende e age.

```
 BORDA                 GATEWAY (FOG)                       NUVEM / SERVIDOR
┌──────────────────┐   ┌──────────────┐                    ┌───────────────────────┐
│ Wearable simulado│   │ Broker MQTT  │                ┌──▶│ Histórico de contexto │
│ (reproduz WESAD) │──▶│ (Mosquitto)  │                │   │ (TimescaleDB)         │
└──────────────────┘   └──────┬───────┘                │   └───────────┬───────────┘
        MQTT                  ▼                        │               ▼
                       ┌──────────────────┐            │   ┌───────────────────────┐
                       │ Ingestão +       │────────────┘   │ Serviço de IA         │
                       │ features (60 s)  │                │ (FastAPI: estado +    │
                       └──────────────────┘                │  previsão)            │
                                                           └───────────┬───────────┘
                                                                       ▼
                                                           ┌───────────────────────┐
                                                           │ Dashboard e alertas   │
                                                           │ (Streamlit, Telegram) │
                                                           └───────────────────────┘
```

| Camada | Tecnologia |
| --- | --- |
| Simulação do sensor | Python + paho-mqtt (opcional: ESP32 no Wokwi para o vídeo) |
| Mensageria | Mosquitto (MQTT) |
| Armazenamento | TimescaleDB ou InfluxDB |
| IA | pandas, NeuroKit2 (features), scikit-learn, XGBoost |
| API | FastAPI |
| Visualização e alerta | Streamlit ou Grafana; bot do Telegram |
| Infra | Docker Compose, GitHub Actions |

---

## 5. Dados e IA

Usamos o **WESAD** (Wearable Stress and Affect Detection, Schmidt et al., ICMI 2018), dataset público para uso acadêmico com sinais reais de 15 participantes em condições rotuladas de repouso, diversão e estresse induzido.

| Sensor | Sinais | Uso no projeto |
| --- | --- | --- |
| Pulseira (Empatica E4) | Pulso (BVP), atividade eletrodérmica (EDA), temperatura, acelerômetro | Principal: é o que um wearable de pulso mede |
| Peitoral (RespiBAN) | ECG, EDA, respiração, temperatura, acelerômetro | Opcional: comparação de desempenho |

Pipeline de IA:

1. **Janelamento:** janelas de 60 s com passo de 5 s a 30 s.
2. **Features:** frequência cardíaca e variabilidade (HRV), nível e picos de EDA, estatísticas de temperatura e movimento.
3. **Classificação do estado:** Random Forest e XGBoost como base; LSTM ou 1D-CNN como comparação, se houver tempo.
4. **Predição de contexto:** o mesmo pipeline treinado para prever o rótulo da janela seguinte, a partir do histórico recente.
5. **Validação:** leave-one-subject-out (treina sem um participante e testa nele), que mede se o modelo generaliza para pessoas novas.
6. **Métricas:** F1 macro, acurácia e matriz de confusão; importância de features para explicar o fenótipo digital.

Referência: o artigo original relata acurácia aproximada de 93% na classificação binária estresse vs. não estresse (valor a confirmar na leitura do artigo).

---

## 6. Informações coletadas

São três fontes: sinais do corpo, contexto e autorrelato.

### 6.1 Sinais fisiológicos (disponíveis no WESAD)

| Sinal | O que indica | Como muda no estresse |
| --- | --- | --- |
| Frequência cardíaca | Ativação do corpo | Sobe |
| Variabilidade da frequência cardíaca (HRV) | Equilíbrio entre "acelerar" e "relaxar" | Cai (marcador mais usado) |
| Atividade eletrodérmica (EDA, suor da pele) | Ativação emocional | Nível sobe e aparecem mais picos |
| Temperatura da pele | Circulação nas extremidades | Tende a cair (mão fria) |
| Respiração (sensor de peito) | Ritmo respiratório | Fica mais rápida e curta |
| Acelerômetro | Movimento | Filtro: coração acelerado durante exercício não é estresse |

### 6.2 Contexto (simulado na demo ou trabalho futuro)

- Horário, dia da semana e qualidade do sono.
- Ambiente, via sensores IoT da sala: ruído, temperatura, CO₂ e luz.
- Agenda, por exemplo reuniões seguidas.

### 6.3 Autorrelato

- Check-ins rápidos ("como você está agora? de 1 a 5"), usados para confirmar alertas e avaliar se a intervenção ajudou.
- No WESAD, os participantes responderam questionários de humor e ansiedade (PANAS e STAI), que ajudam a validar o modelo.

---

## 7. Como identificamos estresse ou mal-estar

1. **Padrão pessoal (fenótipo digital):** os primeiros dias de cada pessoa definem o "normal dela". Os alertas usam o desvio em relação a esse padrão, não um valor fixo.
2. **Estresse agudo:** o modelo classifica cada janela de 60 s. A regra intuitiva é frequência cardíaca acima do normal, HRV em queda e mais picos de EDA, com a pessoa parada.
3. **Mal-estar persistente:** desvios que duram dias (frequência cardíaca de repouso alta, HRV baixa por vários dias, sono ruim). Como o WESAD não tem rótulos para isso, seria uma detecção de anomalias, sem caráter de diagnóstico.
4. **Saída em três níveis:** normal, atenção ou estresse, sempre com a confiança do modelo, para evitar alarmes falsos.

---

## 8. Intervenções

| Nível | Quando | Intervenção |
| --- | --- | --- |
| Antecipar | O modelo prevê estresse na próxima janela | Aviso discreto: "sua tensão está subindo, que tal 2 minutos de pausa?" |
| Agir no momento | Estresse detectado | Respiração guiada lenta (cerca de 6 respirações por minuto, por 2 a 3 min), pausa ativa, alongamento, água |
| Agir no ambiente | Contexto ruim (ruído, CO₂ alto, calor) | Sugerir ventilar ou mudar de sala; num ambiente inteligente completo, acionar ventilação ou ajustar a luz |
| Prevenir | Padrões recorrentes no histórico | Resumo semanal ("seus picos aparecem às terças à tarde, após reuniões seguidas") com sugestão de pausas na agenda |
| Encaminhar | Estresse alto por vários dias ou sinais de mal-estar | Recomendar procurar um profissional de saúde, sem diagnosticar; incluir o contato do CVV (188) na mensagem de apoio |

Regras de boa prática:

- Limite de alertas por dia e horário de silêncio, para evitar que os avisos sejam ignorados.
- Botão "isso ajudou?" após cada intervenção. Com esse retorno, o sistema pode aprender qual intervenção funciona para cada pessoa (possível ligação com Aprendizado por Reforço).

---

## 9. Escopo: demo vs. trabalhos futuros

| Na demo (até 30/11) | No relatório, como trabalho futuro |
| --- | --- |
| Sinais do WESAD com filtro de movimento | Detecção de mal-estar persistente |
| Contexto simulado (horário e ruído) | Sensores de ambiente reais e integração com agenda |
| Alerta antecipado e respiração guiada no painel ou no Telegram | Personalização das intervenções com aprendizado por reforço |
| Botão de retorno "isso ajudou?" | Coleta com dispositivo real e base própria |

---

## 10. Engenharia de Software

A qualidade do código vale 40% da nota, então o projeto é tratado como um produto pequeno, não como um notebook isolado.

- **Arquitetura documentada:** diagrama C4 (contexto e containers) no relatório e no README.
- **Serviços separados:** simulador, ingestão, inferência e dashboard em containers, subindo com um único `docker compose up`.
- **Testes:** pytest para extração de features, contrato das mensagens MQTT e API de inferência.
- **CI:** GitHub Actions rodando lint e testes a cada push.
- **Decisões registradas:** ADRs curtos (por que MQTT, por que esse banco, por que esse modelo).
- **Reprodutibilidade:** script que baixa o WESAD, treina e salva o modelo; versões fixadas no `requirements.txt`.

---

## 11. Ética e LGPD

Dados fisiológicos são dados pessoais sensíveis de saúde (LGPD, art. 5º, II).

- O WESAD já é anonimizado e liberado para uso acadêmico; não coletamos dados de pessoas.
- O sistema é de apoio ao bem-estar, não de diagnóstico; os alertas deixam isso claro.
- Se o grupo usar dados próprios na demo, é preciso consentimento por escrito.
- Num cenário real: consentimento explícito, minimização (enviar só features, não sinal bruto), processamento na borda quando possível e criptografia no MQTT (TLS).

---

## 12. Plano de trabalho

| Período | Fase | Entregas |
| --- | --- | --- |
| 21–22/11 | Definição | Tema aprovado, papéis definidos, repositório criado |
| 23–25/11 | Construção | Simulador, MQTT, banco e modelo v1 |
| 25/11 | Marco | Infraestrutura congelada |
| 26–27/11 | Integração | IA em tempo real, alertas e painel |
| 28/11 | Documentação | Relatório PDF, diagrama C4, README |
| 29/11 | Vídeo | Roteiro, gravação e edição |
| 30/11 | Entrega | Relatório, GitHub e vídeo |

Papéis sugeridos (com 5 ou 6 pessoas, a dupla extra reforça Dados e IA):

| Papel | Responsabilidades |
| --- | --- |
| IoT e infraestrutura | Simulador, broker MQTT, Docker Compose |
| Dados e IA | Features, modelos, validação, métricas |
| Backend e dashboard | Banco, API FastAPI, painel, alertas |
| Qualidade e documentação | Testes, CI, relatório, README, diagrama C4 |
| Pitch e vídeo | Roteiro, gravação e edição (compartilhado por todos) |

---

## 13. Como a proposta atende os critérios de avaliação

| Critério | Peso | Como atendemos |
| --- | --- | --- |
| Complexidade e implementação técnica | 40% | Pipeline completo IoT → banco → IA → ação, em containers, com testes e CI |
| Relevância e inovação | 20% | Predição (não só detecção) e fenótipo digital individual; tema de saúde com dados reais |
| Relatório e documentação | 20% | Seções do enunciado + C4 + ADRs + README com instalação em um comando |
| Vídeo pitch e demo | 20% | História de um usuário: sinais ao vivo, alerta antes do pico, histórico explicado |

---

## 14. Riscos e plano B

| Risco | Plano B |
| --- | --- |
| Predição da próxima janela com desempenho fraco | Entregar a detecção como resultado principal e a predição como experimento discutido |
| Deep learning não fica pronto | Ficar com Random Forest/XGBoost, que já atendem o enunciado |
| Integração Docker atrasa | Congelar a infraestrutura em 25/11 e cortar o Telegram |
| Grupo prefere outro tema | Alternativas na mesma linha: ambiente inteligente na agricultura ou cuidado ubíquo de idosos |

---

## 15. Próximos passos

- [ ] Grupo aprova o tema e define os papéis (até 22/11)
- [ ] Criar repositório público e baixar o WESAD
- [ ] Subir o esqueleto Docker (MQTT + banco + dashboard)
- [ ] Primeiro modelo treinado com métricas leave-one-subject-out

---

## Referências

- SCHMIDT, P. et al. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI)*, 2018.
- Ementas do PPG em Computação Aplicada (Unisinos), 2026/1: Ambientes Inteligentes; Edge, Fog e Cloud; Conectividade Avançada e Sistemas Distribuídos; Inteligência Artificial.
- Lei nº 13.709/2018 (LGPD).
