# 🧠 ScArlet-Sails RAG System

**Retrieval-Augmented Generation для торговых паттернов**

## 📖 Оглавление

1. [Обзор](#обзор)
2. [Архитектура](#архитектура)
3. [Компоненты](#компоненты)
4. [Быстрый старт](#быстрый-старт)
5. [Team Workflow](#team-workflow-для-трейдеров)
6. [API Reference](#api-reference)
7. [Примеры использования](#примеры-использования)
8. [Roadmap](#roadmap)

---

## Обзор

RAG система ScArlet-Sails извлекает и хранит торговые паттерны с полным контекстом:

- **Pattern Extraction**: Автоматическое извлечение 74 features + W_box scoring
- **Time Capsule**: Сохранение snapshot (100 баров до + 50 после)
- **Smart Retrieval**: Поиск похожих паттернов с фильтрацией
- **Council Integration**: Расширенный контекст для принятия решений

### Что хранится?

```
rag/
├── patterns/
│   ├── library.json           # Index of all patterns
│   ├── BTC_1h_*.json          # Pattern metadata
│   └── snapshots/
│       └── BTC_1h_*.csv       # Raw OHLCV data (150 bars)
├── trades/
│   └── trade_log.json         # Historical trades
└── lessons/
    └── lessons.json           # Extracted learnings
```

---

## Архитектура

### Current (v1.0)

```
┌────────────────────────────────────────┐
│          RAG System v1.0               │
└────────────────────────────────────────┘

Layer 1: EXTRACTION (PatternExtractor)
├─ Input: Timestamp + Symbol + Timeframe
├─ Extract: 74 features from "Time Capsule"
├─ Calculate: W_box quality score
├─ Simulate: Multiple TP/SL scenarios
└─ Output: JSON + CSV snapshot

Layer 2: STORAGE (JSON Files)
├─ patterns/library.json (index)
├─ patterns/*.json (metadata)
└─ patterns/snapshots/*.csv (raw data)

Layer 3: RETRIEVAL (RAGRetriever)
├─ Simple JSON loading
├─ Metadata filtering (symbol, tf, direction)
├─ build_council_context() for decisions
└─ Statistics calculation (win rate, avg PnL)

Layer 4: INTEGRATION (Council)
└─ Context for decision making
```

### Planned (v2.0)

```
┌────────────────────────────────────────┐
│    RAG System v2.0 (Multi-HyDE)        │
└────────────────────────────────────────┘

Layer 1: EXTRACTION
└─ [Same as v1.0] ✅

Layer 2: INDEXING (NEW!)
├─ sentence-transformers embeddings
├─ ChromaDB vector database
├─ Metadata filters
└─ Hybrid search

Layer 3: RETRIEVAL (Multi-HyDE)
├─ Query expansion (4 perspectives)
├─ Hypothetical document generation
├─ Semantic search per hypothesis
├─ Reranking (BGE-reranker)
└─ Context building

Layer 4: AUTO-LEARNING (NEW!)
├─ Dexter post-mortem analysis
├─ Automatic pattern extraction
└─ Self-learning loop
```

---

## Компоненты

### 1. PatternExtractor (extractor.py)

**Status:** ✅ Production-ready  
**Version:** 2.0 "Time Capsule"

**Возможности:**
- Извлекает 74 features для конкретного паттерна
- Защита от look-ahead bias (использует бар ДО пробития)
- Box Range metrics (support, resistance, touches)
- W_box quality scoring (4 компонента)
- Future path simulation (тестирует разные TP/SL)
- Сохраняет snapshot (100 баров истории + 50 будущего)

**Использование:**

```python
from rag import PatternExtractor

extractor = PatternExtractor("BTC", "1h")
data = extractor.extract(
    breakout_time="2024-11-26 14:00",
    pattern_type="box_range",
    direction="long",
    lookback=48,
    snapshot_lookback=100,
    snapshot_forward=50
)

if 'error' not in data:
    json_path = extractor.save(data)
    print(f"✅ Saved: {json_path}")
```

---

### 2. RAGRetriever (retriever.py)

**Status:** ✅ Working (basic version)  
**Version:** 1.0

**Методы:**

#### `retrieve_patterns(top_k, filters)`
Простой поиск паттернов с фильтрацией.

```python
from rag import RAGRetriever

retriever = RAGRetriever()

# Get all BTC 1h long patterns
patterns = retriever.retrieve_patterns(
    top_k=5,
    filters={
        'symbol': 'BTC',
        'timeframe': '1h',
        'direction': 'long'
    }
)
```

#### `build_council_context(current_state, top_k)`
**NEW!** Расширенный контекст для Council.

```python
current_state = {
    'symbol': 'BTC',
    'timeframe': '4h',
    'direction': 'long',
    'features': {...}  # Current market features
}

context = retriever.build_council_context(
    current_state=current_state,
    top_k=5
)
```

---

### 3. Config (config.py)

**Status:** ✅ Complete

```python
from rag import COINS, TIMEFRAMES, PATTERNS_DIR, KEY_FEATURES
```

---

### 4. CLI (cli.py)

**Status:** ✅ Working

```bash
# Extract pattern
python -m rag.cli BTC 1h "2024-11-26 14:00" --direction long

# With notes
python -m rag.cli ETH 15m "2024-11-26 09:30" \
    --notes "Strong level, 5 touches"

# Custom lookback
python -m rag.cli SOL 4h "2024-11-25 20:00" --lookback 72

# List all patterns
python -m rag.cli --list

# Statistics
python -m rag.cli --stats
```

---

## Быстрый старт

### Установка

```bash
cd ~/ScArlet-Sails
pip install -r rag/requirements.txt
```

### 1. Извлечь паттерн

```bash
python -m rag.cli BTC 1h "2024-11-26 14:00" --direction long
```

### 2. Использовать в коде

```python
from rag import PatternExtractor, RAGRetriever

# Extract
extractor = PatternExtractor("BTC", "1h")
pattern = extractor.extract("2024-11-26 14:00")
extractor.save(pattern)

# Retrieve
retriever = RAGRetriever()
context = retriever.build_council_context({
    'symbol': 'BTC',
    'timeframe': '1h',
    'direction': 'long'
})

print(f"Win rate: {context['historical_win_rate']}")
print(f"Recommendation: {context['recommendation']}")
```

---

## Team Workflow для трейдеров

### 1. Быстрый старт (5 минут)

```bash
# 1. Перейти в проект
cd ~/ScArlet-Sails

# 2. Установить зависимости
pip install -r rag/requirements.txt

# 3. Проверить что работает
python -m rag.cli --list
# → Покажет: "0 patterns found" (нормально!)
```

---

### 2. Ежедневные команды (Team Routine)

#### Команда A: Pattern Hunters (найти паттерн → извлечь)

```bash
# Найти на TradingView паттерн → записать время пробития
# Например: BTC 1h пробил сопротивление 2024-12-08 14:00

# Извлечь паттерн (30 секунд)
python -m rag.cli BTC 1h "2024-12-08 14:00" --direction long

# С нотами (важно!)
python -m rag.cli ETH 4h "2024-12-08 09:30" \
  --direction long \
  --notes "5 touches resistance, volume spike"

# Проверить что сохранилось
python -m rag.cli --list
# → "Found 3 patterns"
```

#### Команда B: Validators (проверить качество)

```bash
# Посмотреть статистику
python -m rag.cli --stats
# → Win rate, avg W_box, avg PnL по всем паттернам

# Посмотреть детально паттерн
python -m rag.cli BTC 1h "2024-12-08 14:00" --show
# → Покажет W_box=0.65, max_profit=2.3%

# Если W_box < 0.4 → удалить
python -m rag.cli BTC 1h "2024-12-08 14:00" --delete
```

#### Команда C: Council Users (использовать в pipeline)

```bash
# В run_council.py уже интегрировано!
# Просто запускать:
python scripts/run_council.py BTC 4h

# RAG автоматически даст контекст:
# "Found 7 similar patterns, win_rate=68%, confidence=75%"
```

#### Команда D: Data Curators (еженедельная уборка)

```bash
# Посмотреть все паттерны
python -m rag.cli --list --details

# Удалить плохие (W_box < 0.4)
python -m rag.cli --cleanup --min-wbox 0.4

# Backup перед пушем
tar -czf rag_backup_$(date +%Y%m%d).tar.gz rag/patterns/
```

---

### 3. Git workflow (обязательно)

```bash
# Каждый день в 18:00
git add rag/patterns/
git commit -m "Add 5 patterns: BTC_1h_20241208 + ETH_4h_20241207"
git push

# Никогда НЕ коммитить snapshots/*.csv (они огромные!)
# .gitignore уже настроен
```

**Пример коммита:**

```
Add 8 patterns [Dec 8]:
- BTC 1h x3 (W_box: 0.65, 0.72, 0.58)
- ETH 4h x2 (W_box: 0.68, 0.61)  
- SOL 15m x3 (W_box: 0.55, 0.49, 0.63)
Total: 42 patterns in library
```

---

## API Reference

(без изменений, как было выше)

---

## Примеры использования

(без изменений, как было выше)

---

## Roadmap

(без изменений, как было выше)

---

*Last updated: December 8, 2025*
*Version: 1.1 (added Team Workflow section)*
