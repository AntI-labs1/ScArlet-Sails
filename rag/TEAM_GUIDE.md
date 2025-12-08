# 👥 RAG SYSTEM: ПРАКТИЧЕСКОЕ РУКОВОДСТВО ДЛЯ КОМАНДЫ

**Версия:** 1.0  
**Дата:** 8 декабря 2025  
**Для кого:** Вся команда ScArlet-Sails

---

## 🎯 ЧТО ЭТО?

RAG (Память паттернов) — система для:
1. 💾 **Сохранения** торговых паттернов с полным контекстом
2. 🔍 **Поиска** похожих ситуаций по истории
3. 🧠 **Обучения** Council на прошлых сделках

**Главный принцип:** Каждый паттерн — это "Временная капсула" с 74 features + результатом

---

## 🚦 БЫСТРЫЙ СТАРТ (5 минут)

### 1. Проверьте установку

```bash
# Перейти в проект
cd ~/ScArlet-Sails

# Проверить Python
python3 --version  # Должно быть 3.9+

# Установить зависимости
pip install -r rag/requirements.txt

# Проверить RAG
python -m rag.cli --help
```

**Ожидаемый результат:**
```
🧲 ScArlet-Sails Pattern Extractor
Usage: python -m rag.cli ...
```

### 2. Проверьте данные

```bash
# Проверить что есть фичи
ls -lh data/features/*.parquet

# Должно показать 14 монет × 4 таймфрейма = 56 файлов
```

**Если нет:**
```bash
git pull  # Скачать данные
```

---

## 📝 ЕЖЕДНЕВНЫЕ ЗАДАЧИ

### Задача 1: Добавить паттерн в RAG

**Когда:** Нашёл интересный паттерн на TradingView

**Шаги:**

```bash
# 1. Запиши время пробития (свеча когда цена вышла из box)
# Пример: BTC пробил 96800 в 14:00 26 ноября

# 2. Запусти извлечение
python -m rag.cli BTC 1h "2024-11-26 14:00" --direction long

# 3. Система покажет:
# ✅ Support: 95120.50
# ✅ Resistance: 96800.75
# ✅ W_box: 0.4480
# ✅ Max profit: 3.2%
# ✅ Saved: rag/patterns/BTC_1h_20241126_1400.json

# 4. Закоммить
git add rag/patterns/
git commit -m "Add BTC 1h long pattern (W_box=0.45)"
git push
```

**Параметры:**
- `--direction long` или `--direction short` — направление
- `--notes "Текст"` — заметки (например, "Сильный MA50, 5 касаний")
- `--lookback 72` — сколько баров назад смотреть box (по умолчанию 48)

---

### Задача 2: Посмотреть все паттерны

```bash
# Список всех паттернов
python -m rag.cli --list

# Вывод:
# 1. BTC_1h_20241126_1400 (W_box=0.45, +3.2%)
# 2. ETH_4h_20241125_2000 (W_box=0.62, +2.8%)
# 3. SOL_15m_20241124_0930 (W_box=0.38, -1.1%)
# ...

# Статистика
python -m rag.cli --stats

# Вывод:
# Total patterns: 15
# Win rate: 68%
# Avg W_box: 0.54
# Avg PnL: +2.1%
```

---

### Задача 3: Найти похожие паттерны

**В Python коде:**

```python
from rag import RAGRetriever

retriever = RAGRetriever()

# Найти все BTC 4h long
patterns = retriever.retrieve_patterns(
    top_k=10,
    filters={
        'symbol': 'BTC',
        'timeframe': '4h',
        'direction': 'long'
    }
)

print(f"Found {len(patterns)} patterns")

# Статистика
wins = sum(1 for p in patterns if p['future_path']['max_profit_pct'] > 0)
print(f"Win rate: {wins/len(patterns):.1%}")
```

---

### Задача 4: Использовать в Council

**В `scripts/run_council.py` уже интегрировано!**

```python
# Council автоматически получает:
context = retriever.build_council_context(
    current_state={
        'symbol': 'BTC',
        'timeframe': '4h',
        'direction': 'long',
        'features': {...}  # Текущие индикаторы
    },
    top_k=5
)

# context содержит:
# - similar_patterns: 5 похожих паттернов
# - historical_win_rate: 0.68 (на похожих 68% win rate)
# - recommendation: "Strong long setup"
# - confidence: 0.75
```

**Запуск Council:**
```bash
python scripts/run_council.py --coin BTC --timeframe 4h

# Council покажет:
# 🧠 RAG Context:
#    • Found 5 similar patterns
#    • Historical win rate: 68%
#    • Average PnL: +2.3%
#    • Recommendation: Strong long
#    • Confidence: 75%
```

---

## 🛠️ ПОЛЕЗНЫЕ КОМАНДЫ

### Извлечение паттернов

```bash
# Базовая команда
python -m rag.cli BTC 1h "2024-11-26 14:00" --direction long

# С заметками
python -m rag.cli ETH 4h "2024-11-25 20:00" \
    --direction short \
    --notes "Resistance rejection, 3rd touch"

# Длинный lookback (72 бара вместо 48)
python -m rag.cli SOL 4h "2024-11-24 16:00" --lookback 72

# Все монеты доступны:
# BTC, ETH, SOL, AVAX, ALGO, DOT, ENA, HBAR, LDO, LINK, LTC, ONDO, SUI, UNI

# Все таймфреймы:
# 15m, 1h, 4h, 1d
```

### Просмотр

```bash
# Список всех
python -m rag.cli --list

# Статистика
python -m rag.cli --stats

# Помощь
python -m rag.cli --help
```

### Git работа

```bash
# Обновить проект (каждое утро!)
cd ~/ScArlet-Sails
git pull

# Добавить новые паттерны
git add rag/patterns/
git status  # Проверить что добавляется

# Коммит
git commit -m "Add 3 BTC patterns (W_box > 0.5)"

# Отправить
git push
```

---

## ❓ FAQ (ЧАСТЫЕ ВОПРОСЫ)

### Q: Как найти время пробития?

**A:** На TradingView:
1. Найди свечу которая **ЗАКРЫЛАСЬ ВЫШЕ resistance** (для long)
2. Наведи курсор на эту свечу
3. Скопируй время в формате `"YYYY-MM-DD HH:MM"`

**Пример:** `"2024-11-26 14:00"`

**Важно:** Время должно быть **UTC** (если TradingView показывает локальное, переведи).

---

### Q: Что такое W_box?

**A:** Качество паттерна (0.0 - 1.0):

```
W_box = I_rsi × I_volatility × I_volume × I_touches

Где:
- I_rsi: RSI в норме (0.3 - 1.0)
- I_volatility: Низкая волатильность (0.0 - 1.0)
- I_volume: Высокий объём (0.3 - 1.0)
- I_touches: Касания support/resistance (0.3 - 1.0)
```

**Интерпретация:**
- W_box > 0.6 — 🟢 Отличный паттерн
- W_box > 0.4 — 🟡 Хороший
- W_box > 0.3 — 🟠 Средний
- W_box < 0.3 — 🔴 Слабый (не сохранять)

---

### Q: Ошибка "Бар не найден"?

```bash
ValueError: Бар не найден. Ближайший: 2024-11-26 14:15
```

**Причины:**
1. Время не точно совпадает со свечой
2. Свеча отсутствует в данных

**Решение:**
```bash
# Используй ближайшее время
python -m rag.cli BTC 1h "2024-11-26 14:15" --direction long
                               ↑↑↑↑↑ Время из ошибки
```

---

### Q: library.json пустой, это нормально?

**A:** Да! Это начальное состояние.

**Вы сейчас собираете паттерны вручную.** Каждый `python -m rag.cli` добавляет новую запись.

**Через 2 недели:** будет 50-100+ паттернов.

---

### Q: Как часто добавлять паттерны?

**A:** Рекомендация:
- **2-3 паттерна в день** (если находишь на TradingView)
- **После каждой закрытой сделки** (в будущем автомат)

**Цель:** 50-100 паттернов за 2 недели.

---

### Q: Можно удалить паттерн?

**A:** Да, просто удали файл:

```bash
# Найти паттерн
ls rag/patterns/*.json

# Удалить
rm rag/patterns/BTC_1h_20241126_1400.json
rm rag/patterns/snapshots/BTC_1h_20241126_1400.csv

# Коммит
git add rag/patterns/
git commit -m "Remove low quality pattern"
git push
```

---

## 🚨 ЧТО НЕ НАДО ДЕЛАТЬ

### ❌ НЕ редактировать JSON вручную

```bash
# ❌ ПЛОХО:
vim rag/patterns/BTC_1h_*.json  # Можно сломать формат

# ✅ ХОРОШО:
# Если ошибка в паттерне → удали и извлеки заново
rm rag/patterns/BTC_1h_20241126_1400.json
python -m rag.cli BTC 1h "2024-11-26 14:00" --direction long
```

### ❌ НЕ коммитить CSV snapshots отдельно

```bash
# ❌ ПЛОХО:
git add rag/patterns/snapshots/*.csv

# ✅ ХОРОШО:
git add rag/patterns/  # Добавит всё вместе (JSON + CSV)
```

### ❌ НЕ использовать локальное время

```bash
# ❌ ПЛОХО:
python -m rag.cli BTC 1h "2024-11-26 18:00+04:00"  # Локальное время

# ✅ ХОРОШО:
python -m rag.cli BTC 1h "2024-11-26 14:00"  # UTC!
```

---

## 📅 ЕЖЕДНЕВНЫЙ WORKFLOW

### Утро (5 минут)

```bash
cd ~/ScArlet-Sails
git pull  # Обновить проект

# Проверить что добавила команда
python -m rag.cli --stats

# Вывод:
# Total patterns: 23 (+5 from yesterday)
# Win rate: 65%
# Avg W_box: 0.52
```

### Днём (при нахождении паттерна)

```bash
# 1. Нашёл паттерн на TradingView
# 2. Записал время пробития

python -m rag.cli BTC 1h "2024-11-26 14:00" \
    --direction long \
    --notes "MA50 bounce, RSI 32"

# 3. Проверил W_box ≥ 0.3 → сохранил
```

### Вечер (10 минут)

```bash
# Коммит всех паттернов за день
cd ~/ScArlet-Sails
git add rag/patterns/
git status  # Проверить

git commit -m "Add 3 patterns: BTC 1h, ETH 4h, SOL 15m"
git push
```

---

## 📊 ЦЕЛИ НА 2 НЕДЕЛИ

### Неделя 1 (8-14 декабря)

**Цель:** 30 паттернов

```
☐ 10 BTC паттернов (1h, 4h)
☐ 10 ETH паттернов (1h, 4h)
☐ 5 SOL паттернов
☐ 5 другие altcoins
```

### Неделя 2 (15-21 декабря)

**Цель:** +30 паттернов (total 60)

```
☐ 15 BTC паттернов
☐ 15 другие монеты
```

### Критерии качества

- ✅ W_box ≥ 0.3
- ✅ Разнообразие: все 14 монет
- ✅ Разнообразие: все 4 таймфрейма
- ✅ 50% long + 50% short
- ✅ Заметки для каждого

---

## 🚀 ЧТО ДАЛЬШЕ?

**После 2 недель** (когда будет 50-100 паттернов):

1. **Vector Database** — semantic search
2. **Multi-HyDE** — +11% accuracy
3. **Auto-population** — автоматическое добавление после сделок

См. **IMPROVEMENT_PLAN.md** для деталей.

---

## 📞 ПОМОЩЬ

**Если что-то не работает:**

1. 💬 Чат проекта
2. 👤 STAR_ANT (лично)
3. 📝 [README.md](./README.md) — полная документация
4. 🗺️ [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md) — roadmap

---

**Успехов в сборе паттернов!** 🎉

*Last updated: December 8, 2025*
