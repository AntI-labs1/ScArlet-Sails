# Habr / VC.ru пост (русский)

## Заголовок

```
Почему я закрыл 8-месячный проект алготрейдинга: честный отрицательный результат и 4 бага, которые завысили Sharpe в 6 раз
```

Альтернативы:
- `8 месяцев на алготрейдинг. Sharpe оказался завышен в 6 раз. Закрыл проект.`
- `Честный negative-result аудит retail алготрейдинга: 131 walk-forward окно, no edge`

## Tags

- алгоритмическая торговля
- квантовый трейдинг
- python
- backtesting
- vectorbt
- криптовалюты
- золото
- machine learning
- статистический анализ
- open source

## Body

```markdown
**TL;DR**: 8 месяцев строил retail алготрейдинговую систему. После честного multi-market walk-forward аудита (131 окно на 14 криптовалютах и 4 драгметаллах) обнаружил: rule-based технические стратегии не имеют generalizable edge. Заявленный walk-forward Sharpe 2.91 оказался артефактом 4 инфраструктурных багов. Реальный Sharpe после фиксов: 0.4-0.6 — на уровне пассивного 60/40 портфеля, который даёт тот же результат с нулевыми трудозатратами.

Публикую код, статью, baseline и переходный план на пассивный портфель в open-source.

---

# Что было построено

В начале 2025, под впечатлением волны "AI trading agent" релизов (TradingAgents 71k звёзд за год, FinGPT, FinRobot), я начал строить retail-систему. План был амбициозный:

- Три quant-стратегии (rule-based RSI/Bollinger/MA, XGBoost ML, hybrid Q-learning)
- LLM "Council" — несколько AI-агентов спорят о каждом решении
- RAG-слой для retrieval по историческим рыночным состояниям
- Dynamic position sizing с учётом regime/OOD/dispersion
- GARCH volatility models, CVaR risk penalty
- Human-in-the-loop финальное решение

К апрелю 2026: ~10 000 строк Python, 13-страничный ROADMAP, README с заявленными метриками walk-forward Sharpe 2.91, Calmar 8.98, MaxDD -15.4%.

---

# Что пошло не так

В ходе аудита с Claude Opus 4.7 за час нашли 4 инфраструктурных бага, которые **вместе создавали** всю заявленную производительность:

**Баг 1: Look-ahead bias в normalization**

`StandardScaler.fit_transform()` вызывался на всём датасете включая будущие бары. Каждый backtest неявно использовал информацию недоступную на момент сделки.

**Баг 2: Hardcoded current_drawdown=0.0**

Killswitch для риска получал константу 0.0 с комментарием `# TODO`. Никогда не срабатывал, как бы плохо ни шёл портфель.

**Баг 3: Инвертированная формула dispersion → position**

Система **должна была** уменьшать позицию при разногласии стратегий. Unit-тесты утверждали обратное. Тесты "падали" месяцами, я был готов "починить" исходник под неправильные тесты.

**Баг 4: Три разных Sharpe annualization**

`sqrt(252×96)`, `sqrt(252×24×4)`, `sqrt(252×24)` — все три использовались параллельно. Первые два арифметически совпадают (24192), но концептуально разные. Ни один не подходит для крипты 24/7.

Ни один баг не ловится typical unit-тестами. Все четыре пойманы только **out-of-sample прогоном на нескольких рынках**.

---

# Эмпирические результаты после фиксов

## Криптовалюты — mean-reversion

14 монет × 4-часовой TF × 8 walk-forward окон = **108 точек**:
- Средний Sharpe: **−0.70**
- Положительный Sharpe: 38 окон из 108 (35%, **хуже монетки**)

На 15-минутном TF combined strategy потеряла **97% капитала** на 13 из 14 монет за 2.5 года. ~1300 сделок × 0.3% round-trip cost = 405% drag — никакой signal не переживёт.

## Металлы — trend-following

200-day SMA на 4 металлах × 25 лет:
- Средний Sharpe: **+0.44** (все 4 положительные)
- **Соответствует литературе** (Hurst/Ooi/Pedersen "A Century of Evidence")
- **НЕ побеждает buy-and-hold**: золото стратегия +515% против B&H +1629%

Trend-following даёт снижение drawdown ценой более низкой доходности. Это не альфа, это другой profile риска.

## Dual momentum

Antonacci dual momentum на 4 металлах, ежемесячная ребалансировка:
- Стратегия: CAGR 12.70%, Sharpe **0.62**, MaxDD -49.5%
- B&H equal-weight: CAGR 12.09%, Sharpe **0.63**, MaxDD -53.9%

Разница в Sharpe **статистически неотличима от нуля**. Стратегия добавляет risk пропорционально return.

---

# Институциональный reality check

| | Реализованный Sharpe |
|---|---|
| AQR Style Premia Alternative Fund (институциональный multi-factor) | **0.41 since inception** (target был 0.70) |
| SG CTA Index (крупнейшие профессиональные trend-следователи) | **~0.56 long-run** |
| Пассивный 60/40 SPY+TLT, ежеквартальная ребалансировка | **~0.6-0.7** с нулевой работой |
| Мой retail multi-factor | **0.4-0.6** в лучшем случае |

Если **элитные хедж-фонды** с proprietary execution, миллиардами AUM и PhD-квантами реализуют Sharpe 0.4-0.7 на этом же классе стратегий — retail с yfinance **математически не получит 1.0+**.

Это не вопрос усилий. Это **физический потолок класса стратегий** на этих рынках.

---

# Главное наблюдение про AI

Я строил overengineered архитектуру **с помощью** AI. AI был полон энтузиазма: генерировал диаграммы, ROADMAP'ы, design documents. **Никогда не сказал** "это уже решено лучше через `pysystemtrade`" или "литература говорит Sharpe ceiling = 0.5".

Тот же AI потом помог провести аудит. Нашёл баги. Прогнал честный backtest. **Заставил** провести разговор про negative result.

**Паттерн**: AI-assisted construction + AI-assisted audit. У construction фазы нет натурального механизма остановки. У audit фазы — её нужно явно запросить.

Если ты строишь с AI и не запрашиваешь аудит — проект растёт вечно в неправильном направлении.

---

# Honest base rate retail алготрейдинга

По данным AllocateSmartly tracker, AQR data, Dalbar reports, и публичных live-результатов quant-newsletter авторов:

- **2-5%** retail трейдеров пробующих multi-factor систему достигают Sharpe > 0.7 live на 3+ года
- **30-40%** заканчивают чем-то эквивалентным 60/40 в более сложной обёртке (zero dollar alpha vs passive)
- **60-70%** забрасывают за 12-24 месяца

Если ты в median группе — **expected dollar alpha vs passive ≈ 0**. После налогов и transaction costs может быть отрицательный.

---

# Что я сделал вместо "ещё одной попытки"

Закрыл актively-managed trading проект. Capital пошёл в passive 60/40 split с quarterly rebalance. 5 минут работы в квартал.

10 недель, которые я собирался потратить на "multi-factor extension", ушли на эту статью и open-source release.

Сплю лучше.

---

# Что в репозитории (MIT-license)

🔗 https://github.com/StarDust1508/ScArlet-Sails

- `POST_MORTEM.md` — главный документ закрытия проекта
- `paper/drafts/main.md` — академический paper draft (5500 слов, 19 references)
- `paper/drafts/medium.md` + `medium_ru.md` — Medium-адаптированные версии (EN + RU)
- `paper/notebooks/stats.py` — open-source реализация Deflated Sharpe Ratio + PBO (numpy-only)
- `paper/notebooks/missing_backtests.ipynb` — Kaggle-репродукция за 30 минут
- `paper/notebooks/figures.ipynb` — 6 publication-quality фигур
- `paper/results/` — 6 JSON с реальными данными backtest'ов
- `passive/` — портфельная аллокация framework (60/40, All-Weather, Permanent + RU варианты)
- `backtesting/vbt_engine.py` — vectorbt-based backtest engine
- `scripts/fetch_binance_klines.py` + `fetch_metals.py` — бесплатные data fetchers

Открытый код, открытые данные, открытые ошибки. Полная reproducibility.

---

# Если ты рассматриваешь похожий проект

Я **не говорю** "не делай". Образовательная ценность строительства, валидации и **честного провала** огромна. Я узнал больше за 8 месяцев чем за годы чтения финансов.

Но иди с **честным base rate**:

1. **Прочитай 3 книги сначала**: Andreas Clenow "Following the Trend", Gary Antonacci "Dual Momentum", Marcos López de Prado "Advances in Financial Machine Learning". 6 недель чтения, $80.

2. **Не пересоздавай существующее**. Fork `pysystemtrade` Роба Карвера (ex-AHL квант, 11 лет активной разработки). Эта работа уже сделана.

3. **Определи stop-loss для самого проекта** до старта. "Закрою если walk-forward Sharpe через 3 месяца < X". Без этого ты будешь работать ad infinitum.

4. **Тестируй cross-market с первого дня**. Не одна монета. Не один TF. **Multi-asset = единственная честная валидация**.

5. **Используй Deflated Sharpe** для любой опубликованной цифры. Если не можешь сказать "deflated SR X.X over Y trials" — у тебя нет результата, у тебя in-sample артефакт.

---

# Ссылки

- Репозиторий: https://github.com/StarDust1508/ScArlet-Sails
- Полный POST_MORTEM: в репо
- Академическая статья (preprint): arxiv.org/abs/[ID] (после публикации)
- English version этого поста: medium.com/[link]

---

Если откликнулось — буду благодарен за **ваши** честные negative results. Их сложнее найти чем success stories, но они полезнее всем нам.

**60/40 портфель скучный. 60/40 портфель тихо побеждает.**
```

## Метаданные для Habr

- Хабр / VC.ru категория: **Разработка**, подкатегория **Python / Машинное обучение / Финансовое моделирование**
- Хаб: "Алгоритмы", "Python", "Машинное обучение", "Финансы для гиков"
- Сложность: средняя
- Время чтения: ~12 минут

## Tips для Habr-аудитории

- Habr любит **технические детали** — оставь все цифры и формулы
- Habr **не любит** marketing hype — наоборот, **самокритика приветствуется**
- Не используй emoji в основном тексте — выглядит непрофессионально
- Best posting time: будний день 10-11 утра МСК

## После публикации

- Мониторить комментарии первые 4 часа (Habr алгоритм)
- Отвечать на технические вопросы развёрнуто
- Не вступать в флейм с "ну а вот моя стратегия работает"
- Через 48 часов — кросс-пост на VC.ru с лёгкой адаптацией
