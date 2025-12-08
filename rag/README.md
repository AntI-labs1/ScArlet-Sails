# 🧠 ScArlet-Sails RAG System

**Retrieval-Augmented Generation для торговых паттернов**

## 📖 Оглавление

1. [Обзор](#обзор)
2. [Архитектура](#архитектура)
3. [Компоненты](#компоненты)
4. [Быстрый старт](#быстрый-старт)
5. [API Reference](#api-reference)
6. [Примеры использования](#примеры-использования)
7. [Roadmap](#roadmap)

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

**Output structure:**

```json
{
  "id": "BTC_1h_20241126_1400",
  "version": "2.0",
  "meta": {...},
  "box": {
    "support": 95120.50,
    "resistance": 96800.75,
    "box_range_pct": 1.76,
    "touches_support": 4,
    "touches_resistance": 3
  },
  "indicators_before": {
    "rsi_zscore": -0.23,
    "macd_zscore": 0.45,
    ...
  },
  "w_box": {
    "I_rsi": 1.0,
    "I_volatility": 0.8,
    "I_volume": 0.8,
    "I_touches": 0.7,
    "W_box": 0.4480
  },
  "future_path": {
    "max_profit_pct": 3.2,
    "max_drawdown_pct": -0.8,
    "simulations": {...}
  },
  "snapshot": {
    "file": "BTC_1h_20241126_1400.csv"
  }
}
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

# Returns:
# {
#   'similar_patterns': [...],          # Top-k похожих паттернов
#   'historical_win_rate': 0.68,        # Win rate на похожих
#   'avg_pnl': 2.3,                     # Средний PnL
#   'avg_w_box': 0.62,                  # Средний W_box
#   'recommendation': 'Strong long',     # Рекомендация
#   'confidence': 0.75,                 # Уверенность
#   'lessons': [...]                    # Уроки
# }
```

#### Backward compatibility
Старые методы работают:
```python
patterns = retriever.retrieve_similar_patterns(limit=5)
trades = retriever.retrieve_historical_trades(limit=10)
lessons = retriever.get_lessons(limit=3)
```

---

### 3. Config (config.py)

**Status:** ✅ Complete

**Константы:**

```python
from rag import COINS, TIMEFRAMES, PATTERNS_DIR, KEY_FEATURES

# 14 монет
COINS = ["BTC", "ETH", "SOL", "AVAX", ...]

# 4 таймфрейма
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# 74 features grouped
KEY_FEATURES = {
    "price": ["open", "high", "low", "close", "volume"],
    "normalized": ["norm_rsi_zscore", ...],
    "regime": ["regime_rsi_low", ...],
    "divergence": ["div_rsi_bullish", ...],
    "time": ["time_hour", ...]
}
```

**Helpers:**

```python
from rag.config import get_file_path

# Get path to features parquet
path = get_file_path("BTC", "1h")
# Returns: data/features/BTC_USDT_1h_features.parquet
```

---

### 4. CLI (cli.py)

**Status:** ✅ Working

**Команды:**

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

### 3. Интеграция с Council

```python
# В scripts/run_council.py
from rag import RAGRetriever

retriever = RAGRetriever(use_multi_hyde=False)
context = retriever.build_council_context(current_state, top_k=5)

# context содержит:
# - similar_patterns: похожие паттерны
# - historical_win_rate: win rate
# - recommendation: рекомендация
# - confidence: уверенность
```

---

## API Reference

### PatternExtractor

```python
class PatternExtractor:
    def __init__(self, coin: str, timeframe: str)
    
    def extract(
        self,
        breakout_time: str,           # "YYYY-MM-DD HH:MM"
        pattern_type: str = "box_range",
        direction: str = "long",      # "long" | "short"
        lookback: int = 48,           # Bars for box
        snapshot_lookback: int = 100, # Bars before
        snapshot_forward: int = 50,   # Bars after
        notes: str = ""
    ) -> Dict
    
    def save(self, data: Dict) -> Path
```

### RAGRetriever

```python
class RAGRetriever:
    def __init__(
        self, 
        rag_root: str = "./rag",
        use_multi_hyde: bool = False  # v2.0 feature
    )
    
    def retrieve_patterns(
        self,
        top_k: int = 5,
        filters: Optional[Dict] = None  # {symbol, timeframe, direction}
    ) -> List[Dict]
    
    def build_council_context(
        self,
        current_state: Dict,  # {symbol, timeframe, direction, features}
        top_k: int = 5
    ) -> Dict  # {similar_patterns, win_rate, recommendation, confidence, ...}
    
    # Backward compatibility
    def retrieve_similar_patterns(self, limit: int = 5) -> List[Dict]
    def retrieve_historical_trades(self, limit: int = 10) -> List[Dict]
    def get_lessons(self, limit: int = 5) -> List[Dict]
```

---

## Примеры использования

### Scenario 1: Ручное добавление паттерна

```python
from rag import PatternExtractor

# 1. Найдёшь на TradingView паттерн
#    Время пробития: 2024-11-26 14:00

# 2. Извлечь
extractor = PatternExtractor("BTC", "1h")
pattern = extractor.extract(
    breakout_time="2024-11-26 14:00",
    direction="long",
    notes="Strong MA50 bounce, 5 touches"
)

# 3. Сохранить
if 'error' not in pattern:
    path = extractor.save(pattern)
    print(f"Saved: {path}")
    print(f"W_box: {pattern['w_box']['W_box']}")
    print(f"Max profit: {pattern['future_path']['max_profit_pct']}%")
```

### Scenario 2: Поиск похожих паттернов

```python
from rag import RAGRetriever

retriever = RAGRetriever()

# Поиск всех BTC 4h long паттернов
patterns = retriever.retrieve_patterns(
    top_k=10,
    filters={'symbol': 'BTC', 'timeframe': '4h', 'direction': 'long'}
)

# Статистика
win_rate = sum(1 for p in patterns 
               if p['future_path']['max_profit_pct'] > 0) / len(patterns)
print(f"Win rate: {win_rate:.1%}")

avg_pnl = sum(p['future_path']['max_profit_pct'] for p in patterns) / len(patterns)
print(f"Avg PnL: {avg_pnl:.1f}%")
```

### Scenario 3: Council integration

```python
from rag import RAGRetriever

retriever = RAGRetriever()

# Current market state
current_state = {
    'symbol': 'BTC',
    'timeframe': '4h',
    'direction': 'long',
    'features': {
        'rsi': 28.5,
        'price_to_sma50': -1.2,
        'atr_pct': 2.3,
        'volume_ratio': 0.95
    }
}

# Get enriched context
context = retriever.build_council_context(current_state, top_k=5)

print(f"Found {context['patterns_count']} similar patterns")
print(f"Historical win rate: {context['historical_win_rate']:.1%}")
print(f"Average PnL: {context['avg_pnl']:.1f}%")
print(f"Recommendation: {context['recommendation']}")
print(f"Confidence: {context['confidence']:.1%}")

if context['confidence'] > 0.7:
    print("✅ Strong setup")
else:
    print("⚠️ Weak setup")
```

---

## Структура данных

### Pattern JSON

```json
{
  "id": "BTC_1h_20241126_1400",
  "version": "2.0",
  "created_at": "2024-11-26T18:30:00",
  
  "meta": {
    "coin": "BTC",
    "timeframe": "1h",
    "pattern_type": "box_range",
    "direction": "long",
    "notes": "Strong MA50 bounce"
  },
  
  "timing": {
    "breakout_time_actual": "2024-11-26 14:00:00+00:00",
    "setup_time": "2024-11-26 13:00:00+00:00"
  },
  
  "box": {
    "support": 95120.50,
    "resistance": 96800.75,
    "box_range_pct": 1.76,
    "touches_support": 4,
    "touches_resistance": 3,
    "atr_box": 620.30,
    "duration_bars": 48
  },
  
  "indicators_before": {
    "rsi_zscore": -0.23,
    "macd_zscore": 0.45,
    "atr_zscore": -0.80,
    "volume_zscore": 0.12,
    "div_rsi_bullish": 1,
    "regime_rsi_mid": 1,
    "session_european": 1
  },
  
  "w_box": {
    "I_rsi": 1.0,
    "I_volatility": 0.8,
    "I_volume": 0.8,
    "I_touches": 0.7,
    "W_box": 0.4480
  },
  
  "future_path": {
    "max_profit_pct": 3.2,
    "max_drawdown_pct": -0.8,
    "future_bars": 50,
    "simulations": {
      "TP2.0_SL1.0": {"result": "TP", "exit_bar": 12},
      "TP3.0_SL1.5": {"result": "TP", "exit_bar": 28}
    }
  },
  
  "snapshot": {
    "lookback_bars": 100,
    "forward_bars": 50,
    "total_bars": 151,
    "file": "BTC_1h_20241126_1400.csv"
  }
}
```

### library.json

```json
{
  "version": "1.0",
  "patterns": [
    {
      "id": "BTC_1h_20241126_1400",
      "w_box": 0.4480,
      "outcome": "TP",
      "pnl_pct": 2.1,
      "added_at": "2024-11-26T18:30:00"
    }
  ],
  "last_updated": "2024-11-26T18:30:00"
}
```

---

## Roadmap

### ✅ v1.0 (Current)
- [x] PatternExtractor с Time Capsule
- [x] W_box quality scoring
- [x] Future path simulation
- [x] Simple JSON retrieval
- [x] build_council_context() API
- [x] CLI interface

### 🚧 v1.5 (In Progress)
- [ ] Заполнить library.json (100+ паттернов)
- [ ] trades/trade_log.json
- [ ] lessons/lessons.json
- [ ] Statistics dashboard

### 🔮 v2.0 (Planned)
- [ ] **Vector Database** (ChromaDB)
- [ ] **Embeddings** (sentence-transformers)
- [ ] **Multi-HyDE Retrieval** (+11% accuracy)
- [ ] **Reranker** (BGE-reranker-v2-m3)
- [ ] **Auto-population** (Dexter post-mortem)
- [ ] **LLM Integration** (Qwen2.5-Coder local)

### Dependencies for v2.0

```bash
# Add to requirements.txt
chromadb>=0.4.0
sentence-transformers>=2.2.0
FlagEmbedding>=1.2.0
transformers>=4.35.0
torch>=2.0.0
```

---

## Troubleshooting

### Ошибка: "Файл не найден"

```bash
FileNotFoundError: data/features/BTC_USDT_1h_features.parquet
```

**Решение:**
```bash
cd ~/ScArlet-Sails
git pull  # Обновить данные
```

### Ошибка: "Бар не найден"

```bash
ValueError: Бар не найден. Ближайший: 2024-11-26 14:15
```

**Причины:**
1. Время не UTC
2. Свеча не существует в данных
3. Неправильный формат времени

**Решение:**
- Используй UTC время
- Проверь формат: `"YYYY-MM-DD HH:MM"`
- Проверь что свеча есть в данных

### library.json пустой

**Нормально!** Это начальное состояние.

**Заполнить:**
```bash
# Извлечь 5 паттернов
python -m rag.cli BTC 1h "2024-11-26 14:00"
python -m rag.cli ETH 4h "2024-11-25 20:00"
python -m rag.cli SOL 15m "2024-11-24 09:30"
...

# Проверить
python -m rag.cli --list
```

---

## Contributing

### Добавить паттерн

1. Найди на TradingView
2. Запиши время пробития
3. `python -m rag.cli COIN TF "TIME"`
4. Закоммить:

```bash
git add rag/patterns/
git commit -m "Add pattern: COIN_TF_DATE"
git push
```

### Улучшить код

1. Fork репозитория
2. Создай feature branch
3. Реализуй улучшение
4. Pull request

---

## License

Part of ScArlet-Sails project.

---

## Contact

Вопросы → Чат проекта или STAR_ANT

---

*Last updated: December 8, 2025*
*Version: 1.0*
