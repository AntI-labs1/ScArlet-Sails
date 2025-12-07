# ScArlet-Sails Roadmap

## Vision

Систему, где quant-стратегии дают численные сигналы, LLM Council интерпретирует контекст, а человек принимает финальное решение.

**Зачем:**
Не найти "идеальный edge", а построить процесс принятия решений, который:
- учится на своих ошибках,
- контролирует риск,
- генерирует данные для исследований.

**Философия:**
Council — не автопилот, а "вторая голова". Человек несёт риск и ответственность. Система помогает видеть то, что человек может пропустить.

---

## Success Definition

### Через 6 месяцев — три слоя успеха:

**1. Практика (P&L):**
- Paper trading показывает стабильную положительную доходность
- Дневная просадка <= заданного лимита
- Предсказуемое поведение, не "один раз выстрелили"

**2. Research:**
- Датасет решений: P_rb, P_ml, P_hyb, Council, человек, исходы
- Dispersion analysis с p-value
- Черновик выводов уровня технического отчёта

**3. Система:**
- Команда работает по процессу
- RAG пополняется системно
- Есть каденция: ретро, разбор ошибок, улучшение

### "Не зря" — минимум:
- Архитектура задокументирована
- Датасет: S(t), решения, исходы
- Прокачан процесс принятия решений

---

## Current State

### Готово ✅

**Инфраструктура:**
- 59 parquet файлов (14 монет × 4 TF)
- Feature engine v2 (74 признака)
- DVC версионирование

**Quant:**
- Rule-Based Strategy (P_rb)
- XGBoost ML Strategy (P_ml)
- OpportunityScorer, AdvancedRiskPenalty

**Council каркас:**
- CanonicalState (core/canonical_state.py)
- RAG Retriever (rag/retriever.py)
- BaseAgent (council/base_agent.py)

**Документация:**
- README.md, ARCHITECTURE.md
- Configs (council.yaml, risk_limits.yaml)

### Не готово ❌

- Hybrid Strategy (P_hyb)
- Council contracts (Stage 0/1/2/3)
- Quant Aggregator
- Human Interface
- Position sizing по формуле
- Kill-switch
- Dispersion analysis module

---

## Phases

### Phase 1: One Candle Decision

**Цель:** Один сквозной путь от данных до решения.
```
S(t) → CanonicalState → Quant Signals → Council → Human Decision → RAG Log
```

**Задачи:**

1.1 **Council Contracts**
- Создать council/contracts.py
- Stage 0: CouncilContext (snapshot + constraints)
- Stage 1: AgentOpinion (action, confidence, justification)
- Stage 2: AgentReview (peer review, risk flags)
- Stage 3: CouncilRecommendation (final decision)

1.2 **Quant Aggregator**
- Собрать P_rb, P_ml в одну структуру
- Вычислить agreement score
- Определить regime

1.3 **Pattern Detector (Rule-Based)**
- Один агент без LLM
- Использует quant signals + RAG
- Выдаёт AgentOpinion

1.4 **Human Interface (CLI)**
- Показать recommendation
- Accept / Modify / Reject
- Логировать в RAG

1.5 **End-to-End Test**
- Один проход на BTC/4h
- Все контракты работают
- Решение залогировано

**Gate → Phase 2:**
- [ ] Один полный цикл работает
- [ ] Все компоненты связаны по контрактам
- [ ] Решение залогировано в trade_log.json

---

### Phase 2: Patterns & Data

**Цель:** Набрать данные для статистики.

**Задачи:**

2.1 **Pattern Library (3-5 паттернов)**
- MA50 Bounce
- Oversold Reversal
- Breakout
- По 40-60 примеров каждый

2.2 **Historical States**
- Собрать S(t) из всех parquet
- Разметить outcomes (win/loss)
- Сохранить в rag/states/

2.3 **Pattern Detector (LLM)**
- Подключить локальную LLM или API
- Промпт с контрактом JSON
- Fallback на rule-based

2.4 **Screenshot Integration**
- Генерация графиков из OHLCV
- Vision LLM описание
- Добавить в CouncilContext

**Gate → Phase 3:**
- [ ] 3+ паттернов с 40+ примерами
- [ ] 200+ исторических S(t) с outcomes
- [ ] LLM агент работает (или доказано что не даёт value)

---

### Phase 3: Risk & Hybrid

**Цель:** Полноценный risk management и третья стратегия.

**Задачи:**

3.1 **Position Sizing**
```
position_size = risk_per_trade / SL_distance
```
- Читать из risk_limits.yaml
- Валидировать рекомендации Council
- Enforce max_position_size

3.2 **Kill-Switch**
- Трекинг daily P&L
- Трекинг weekly P&L
- Блокировка при пробое лимитов

3.3 **Hybrid Strategy (P_hyb)**
```
P_hyb = α(t)·P_rb + β(t)·P_ml + γ·V_RL
```
- Адаптивные веса по rolling performance
- RL компонент (или упрощённая оценка)

3.4 **Risk Assessor Agent**
- Второй агент Council
- Фокус на position size и SL/TP
- AgentReview для peer review

**Gate → Phase 4:**
- [ ] Position sizing работает по формуле
- [ ] Kill-switch протестирован
- [ ] P_hyb интегрирован в aggregator
- [ ] 2 агента в Council

---

### Phase 4: Paper Trading

**Цель:** 30 дней симуляции с реальными решениями.

**Prerequisite:**
- Backtest без катастрофических просадок
- Risk management работает
- Kill-switch протестирован

**Задачи:**

4.1 **Market Data**
- Real-time или near-real-time feed
- Обновление CanonicalState

4.2 **Order Simulator**
- Market orders + slippage
- SL/TP monitoring
- Trade logging

4.3 **Daily Reports**
- Trades executed
- Win rate, P&L
- Council accuracy

4.4 **Campaign (30 days)**
- Human reviews daily
- Weekly retrospectives
- Lessons в RAG

**Gate → Phase 5:**
- [ ] 30+ трейдов залогировано
- [ ] Не пробиты лимиты просадки
- [ ] Нет дней с неадекватным поведением
- [ ] Lessons documented

---

### Phase 5: Research & Analysis

**Цель:** Dispersion analysis и выводы.

**Задачи:**

5.1 **Dispersion Module**
- ANOVA test (F-statistic, p-value)
- Kolmogorov-Smirnov test
- Variance decomposition

5.2 **Regime Analysis**
- Accuracy по стратегиям по режимам
- Agreement vs outcome correlation

5.3 **Visualization**
- Heatmaps корреляций
- Scatter plots agreement/outcome
- Equity curves

5.4 **Technical Report**
- Methodology
- Results
- Discussion

**Gate → Production:**
- [ ] Dispersion статистически значим (p < 0.05)
- [ ] Понятно когда какая стратегия сильна
- [ ] Отчёт готов для review

---

### Phase 6: Contrarian & Full Council

**Цель:** Три агента с протоколом дискуссии.

**Задачи:**

6.1 **Contrarian Agent**
- Devil's advocate
- Ищет слабости в reasoning
- Снижает confidence при concerns

6.2 **Discussion Protocol**
- Round 1: Independent opinions
- Round 2: Peer review
- Round 3: Aggregation

6.3 **Monitoring Dashboard**
- Equity curve
- Open positions
- Council decisions log

**Gate → Real Trading:**
- [ ] 3 агента работают вместе
- [ ] Discussion protocol обкатан
- [ ] 2+ месяца paper trading без систематического развала

---

## Risk & Fallback

### Если локальная LLM слабая:
1. Прототипировать на API (ограниченно)
2. Понять есть ли value от LLM
3. Если да — портировать на локальную
4. Если нет — Council остаётся rule-based + RAG + человек

### Если паттерны не дают edge:
- После 100+ сделок суммарно — честная оценка
- Варианты: менять паттерны, менять критерии, сместить акцент
- "Не провал" если остались: инфраструктура, датасет, процессы

### Если paper trading -10% за месяц:
1. Остановить автоматические решения
2. Разбор: по паттернам, по режимам, по типам ошибок
3. Два месяца подряд deep negative → пауза и пересборка

### Council vs Human не согласны:
- В моменте прав человек
- Конфликт логируется
- Ретро-анализ: кто был прав чаще

---

## Team & Process

### Роли:

| Область | Ответственный |
|---------|---------------|
| Архитектура, критичный код | ANT_S |
| Ревью решений Council | ANT_S |
| Аннотация паттернов | Команда проекта |
| Пополнение RAG | Команда проекта |
| Ретро-разборы | Все вместе |
| Финальные решения | ANT_S |

### Процесс:

**Ежедневно (async):**
- Сколько примеров размечено
- Сложные кейсы

**Еженедельно (sync):**
- Какие паттерны работают / не работают
- Повторяющиеся ошибки
- Что добавить в lessons.json

### Где хранится знание:
- **RAG** — опыт (паттерны, сделки, уроки)
- **Docs** — карта (архитектура, протоколы, roadmap)
- **Чат** — операционка (не источник истины)

---

## Metrics

### Паттерн работает:
- 50-100 сделок
- Win rate > 50-55%
- R:R >= 1.5 после комиссий
- Нет провала в конкретных режимах

### Council лучше quant:
- Сравнение на одном периоде
- Не только P&L, но и просадка, волатильность equity
- Меньше "очевидных" ошибок

### Готов к paper trading:
- Backtest без катастроф
- Position sizing по формуле
- Kill-switch работает

### Paper trading успешен:
- 30+ трейдов
- Не пробиты лимиты
- Нет дней полной неадекватности

---

## Principles

1. **Контракты важнее кода.**
   Сначала договориться о форматах, потом реализовывать.

2. **Один проход важнее красоты.**
   End-to-end работает > много компонентов по отдельности.

3. **Данные важнее мнений.**
   Backtest и статистика, не ощущения.

4. **Человек решает, система помогает.**
   Council — augmentation, не автопилот.

5. **Логировать всё.**
   Каждое решение, каждый исход, каждый урок.

6. **Консервативно в риске.**
   Сохранение капитала > накрутка прибыли.

7. **Итерации маленькие.**
   Работающий инкремент лучше большого переписывания.

8. **"Не зря" даже если edge = 0.**
   Архитектура + датасет + процесс = ценность.

---

## Dependencies
```
Phase 1 (One Candle)
    │
    ├── 1.1 Contracts
    ├── 1.2 Quant Aggregator
    ├── 1.3 Pattern Detector (rule-based)
    ├── 1.4 Human Interface
    └── 1.5 End-to-End Test
            │
            ▼
Phase 2 (Patterns & Data)
    │
    ├── 2.1 Pattern Library
    ├── 2.2 Historical States
    ├── 2.3 Pattern Detector (LLM)
    └── 2.4 Screenshots
            │
            ▼
Phase 3 (Risk & Hybrid)
    │
    ├── 3.1 Position Sizing
    ├── 3.2 Kill-Switch
    ├── 3.3 Hybrid Strategy
    └── 3.4 Risk Assessor
            │
            ▼
Phase 4 (Paper Trading)
    │
    ├── 4.1 Market Data
    ├── 4.2 Order Simulator
    ├── 4.3 Daily Reports
    └── 4.4 Campaign (30 days)
            │
            ▼
Phase 5 (Research)
    │
    ├── 5.1 Dispersion Module
    ├── 5.2 Regime Analysis
    ├── 5.3 Visualization
    └── 5.4 Technical Report
            │
            ▼
Phase 6 (Full Council)
    │
    ├── 6.1 Contrarian Agent
    ├── 6.2 Discussion Protocol
    └── 6.3 Monitoring Dashboard
            │
            ▼
        Real Trading
```

---

*Scarlet Sails — racing the market through disciplined process and collective intelligence.*
