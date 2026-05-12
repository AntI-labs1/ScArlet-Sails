# ScArlet-Sails — Project Closure & Post-Mortem (Revision 2026-05)

> **Status**: closed as actively-managed trading project, repurposed as research-baseline + reusable infrastructure.
> **Tag**: `v1.0-research-baseline`
> **Date**: 2026-05-11

---

## TL;DR

8 месяцев работы над retail алготрейдинговой системой. Финальный честный
вывод после **108 walk-forward точек на крипте + 23 точки на металлах**:
**generalizable trading edge не достигнут**. Литературная trend-following
baseline (Sharpe 0.5-0.7 на металлах) подтверждена, но **не превосходит**
пассивный 60/40 портфель, требующий нулевой работы.

Проект **закрыт как тред «активный трейдинг»**, **переоткрыт** как:
1. Источник для research paper (negative result audit)
2. Reusable toolkit (vbt-engine, walk-forward, data fetchers)
3. Архив lessons learned для будущих проектов

---

## Что было построено

### Архитектура (~10 000 LoC)

| Слой | Что работало | Что не работало |
|---|---|---|
| **Backtest engine** | `backtesting/vbt_engine.py` (vectorbt-based, canonical Sharpe) | Семь самописных backtest-петель (deprecated) |
| **Data pipeline** | `core/data_loader.py` + `fetch_binance_klines.py` + `fetch_metals.py` | DVC remote не сконфигурирован, реальные raw данные потерянны |
| **Strategies** | `simple_strategies.py` (RSI/Combined/RuleBased/Hybrid) | Council/RAG/QuantAggregator (~3000 LoC research scaffold, не подключены) |
| **Risk management** | `core/dynamic_position_sizer.py`, `rolling_dispersion.py` | OOD/GARCH/CVaR (~800 LoC, написано, не используется в backtest) |
| **Research** | Walk-forward methodology, canonical bars-per-year (365 для крипты) | Q-learner (577 LoC, deprecated после ревизии) |

### Что осталось как **value asset**

- `backtesting/vbt_engine.py` — production-grade vectorbt wrapper
- `backtesting/MIGRATION_NOTES.md` — мигрирование с самописных backtests
- `core/data_loader.py` — supports BTC_USDT и BTCUSDT naming
- `core/metrics_calculator.py` — canonical bars-per-year per timeframe
- `scripts/fetch_binance_klines.py` — Binance Vision OHLCV fetcher
- `scripts/fetch_metals.py` — yfinance futures fetcher (gold/silver/copper/platinum)
- `tests/conftest.py` — synthetic OHLCV bootstrap для CI/Kaggle

---

## Что было выяснено (главное)

### 1. Rule-based mean-reversion не работает на крипте

**Тесты**: 14 монет (BTC, ETH, SOL, AVAX, DOT, LINK, UNI, LTC, ALGO, HBAR, LDO, SUI, ENA, ONDO) × 4 таймфрейма (15m, 1h, 4h, 1d) × 8 walk-forward окон = **108 точек данных**.

**Результат**: Avg Sharpe **−0.70**, positive windows 38/108 (35%, хуже случайности). 13/14 монет combined strategy потеряла > 90% капитала за 2.5 года на 15m таймфрейме — комиссионная мясорубка.

**Вывод**: на bull-market крипте 2023-2026 простые технические индикаторы не имеют generalizable edge. SOL Sharpe 1.2 в одном тесте — статистический выброс, не сигнал.

### 2. Trend-following на металлах работает, но не бьёт passive

**Тесты**: GOLD/SILVER/COPPER/PLATINUM × 1d × 25 лет.

**Результат**:
- 200d SMA filter: Avg Sharpe **+0.44** (4/4 положительный)
- Dual momentum (12m lookback, top-2): Sharpe **+0.62**, **≈ B&H equal-weight Sharpe 0.63**
- Gold/Silver ratio mean reversion: Sharpe **+0.44**

**Вывод**: literature-baseline подтверждена. Но **edge маржинальный**, **не превосходит** пассивный equal-weight портфель из тех же активов.

### 3. Industry benchmarks говорят о пределах

- **AQR Style Premia Fund** (institutional multi-factor): live Sharpe **0.41** since inception (vs target 0.70)
- **SG CTA Index since 2000** (best-in-class CTAs): Sharpe **0.56**
- **Passive 60/40 SPY+TLT**: Sharpe **~0.6-0.7** с zero work
- **Retail multi-factor attempts**: ~2-5% живут 3+ года с Sharpe > 0.7

**Вывод**: даже элитные хедж-фонды с миллиардами AUM, proprietary данными
и квантами PhD дают Sharpe 0.4-0.7. Retail с yfinance не достанет 1.0+.
Структурный ceiling для retail multi-factor: **gross 0.7-0.8, net 0.4-0.6**.

### 4. Critical infrastructure bugs were lurking

В ходе ревизии 2026-05 найдены **серьёзные баги**, которые незаметно
портили все предыдущие backtest-результаты:

- **Look-ahead bias**: `StandardScaler.fit_transform` молчаливо вызывался на
  полном датасете (train+test) в `feature_engine_v2.py`. Все backtest-результаты
  до ревизии были завышены неизвестно насколько.
- **Killswitch off**: `current_drawdown=0.0` зашит в `scripts/run_council.py`.
  Portfolio risk management не работал.
- **Dispersion logic inverted**: формула давала противоположный multiplier
  (agreement → low position, chaos → high position).
- **Sharpe annualization drift**: 7 разных формул в проекте (252×96 vs 252×24×4
  vs 252×24). Все backtests до унификации были не сопоставимы.
- **Data naming mismatch**: loader ждал `BTC_USDT_15m.parquet`, DVC хранил
  `BTCUSDT_15m.parquet`. 5 тестов падали с FileNotFoundError скрыто.

**Этический вывод**: ни одна из цифр в `README.md` (Sharpe 2.91, Calmar 8.98)
**не была воспроизводимой**. Они были в-сэмпл артефактами с look-ahead.
Это **тот же самообман**, против которого предупреждают Bailey & Lopez de Prado.

---

## Что не было сделано (намеренно)

| Не сделано | Почему |
|---|---|
| Council / RAG / LLM-агенты integration | Архитектура задумана как research-scaffold, реальный edge от LLM не доказан в литературе (StockBench 2025) |
| Live trading / paper trading | Не имеет смысла без устойчивого backtest edge |
| Hybrid Q-learner production | Hand-rolled Q-learning устарел; stable-baselines3 был бы правильный путь, но отменён вместе с trading-ambition'ом |
| Fork of pysystemtrade | Rejected after C+D decision — paper не требует production framework |
| Real BTC data via DVC | DVC remote не сконфигурирован; решено через Binance Vision public archive |

---

## Lessons Learned

### Технические

1. **Walk-forward — обязателен**, не опционален. Single backtest = casino.
2. **Costs доминируют edge** на коротких таймфреймах. 15m trading с 0.15%
   round-trip на крипте = death by 1000 cuts. ≥ 4h на крипте, ≥ 1d на металлах.
3. **In-sample Sharpe деflateн в 2-3 раза при честной валидации** (Bailey & Lopez de Prado).
4. **Один factor → Sharpe ~0.5 max**. Multi-factor добавляет +0.1-0.2 за счёт
   некоррелированности. Sharpe 1.0+ требует pyramid из 5+ uncorrelated edges.
5. **Trend > mean-reversion** на trending assets. Я месяц гонял mean-reversion
   стратегии на bull market — это математически проигрышная комбинация.

### Методологические

1. **Hypothesis must precede backtest**, не наоборот. Cherry-picking из шумного
   тестового пространства = data dredging.
2. **Stop conditions важнее start conditions**. До начала проекта надо
   определить «когда я признаю что не работает». Я не определил, и потратил
   на 4 месяца больше нужного.
3. **Multi-coin / multi-market walk-forward** > single-coin deep dive. Один
   coin с Sharpe 1.2 — это шум. 14 coins с Avg Sharpe −0.7 — это сигнал.
4. **Negative results публикуемы** и полезны сообществу больше чем 100-й
   положительный backtest стратегии-X.

### Про себя

1. **«ИИ усугубил» — правда**. Когда я (как пользователь) описывал AI
   видение проекта, AI с энтузиазмом помогал писать architecture-document'ы
   на 10 страниц. AI **не остановил меня** и не сказал «это уже сделано,
   посмотри TradingAgents». Это **systematic bias** AI-инструментов: они
   рады генерировать, плохи в остановке.
2. **«8 месяцев» != «8 месяцев продуктивной работы»**. Реальной trading-
   валидационной работы было ~10% времени, остальное — architecture
   astronaut'ство (Council, RAG, OOD, GARCH-каркасы которые никогда не запустились).
3. **Roadmap.md писал по-русски, ROADMAP.md по-английски** — это **симптом**
   confusion в собственных целях. Кто аудитория? Research community или
   personal log? До конца было неясно.

---

## Что дальше — двух-track plan

### Track C — Research Paper (3-4 недели)

См. `paper/README.md` для outline и status.

Цель: technical report «Rule-Based Trading Strategies Fail to Generalize:
A Multi-Market Walk-Forward Audit (Crypto 2023-2026 + Metals 2000-2026)».

Targets:
- arXiv preprint (week 4)
- SSRN parallel posting
- ICAIF 2026 workshop submission (если откроется до публикации)
- Medium / Substack adapted version для retail-аудитории

### Track D — Passive Capital Allocation

См. `passive/README.md` для портфельной аллокации.

Решение: capital → passive risk-parity portfolio (60/40 / All Weather /
Permanent Portfolio в зависимости от broker access). Quarterly rebalance
через `passive/rebalance.py`. **Никакого daily trading.**

---

## How to use this repo going forward

```
ScArlet-Sails/
├── backtesting/          ← USE: vbt_engine.py for any future backtest
├── core/                  ← USE: data_loader, metrics_calculator
├── scripts/               ← USE: fetch_binance_klines.py, fetch_metals.py
├── paper/                 ← ACTIVE: research paper development
├── passive/               ← ACTIVE: real capital allocation
│
├── strategies/            ← REFERENCE ONLY: rule-based не работают
├── council/               ← REFERENCE ONLY: research scaffold не подключён
├── rag/                   ← REFERENCE ONLY: research scaffold не подключён
├── analysis/              ← DEPRECATED: старые backtest-скрипты
├── archive/               ← REMOVED in revision 2026-05
└── tests/                 ← USE: conftest.py для CI, остальное per-component
```

---

## Acknowledgments

Проект бы не достиг этой точки без жёсткой honest сессии с Claude (Opus 4.7)
в ревизии 2026-05. AI-ассистент сделал то что должен был сделать человек:
посмотрел на 8 месяцев работы и сказал «ваш Sharpe 2.91 в README — это
look-ahead bias, ваш RSI на 15m математически не может зарабатывать,
вернёмся к развилке». **Это саркастично что AI помог remediate то, что
AI же помог наплодить** — но эта pattern полезна для следующих проектов.

---

## License & Reproducibility

Все материалы (код + данные + paper draft) — MIT License.
Reproducibility: см. `kaggle/smoke_test.ipynb` и `tests/conftest.py`.
Контакт: [issue на GitHub repo]
