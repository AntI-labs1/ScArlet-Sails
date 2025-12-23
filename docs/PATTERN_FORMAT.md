# Pattern Format Specification

## Overview

Паттерны — это структурированные описания торговых ситуаций для RAG системы.
Каждый паттерн описывает: контекст входа, действие, результат.

**Philosophy:**
> "Паттерны — это память системы. Хорошо описанный паттерн = хорошее решение в будущем."

---

## Quick Start

### Минимальный паттерн

```json
{
  "id": "ma50_bounce_001",
  "name": "MA50 Bounce",
  "direction": "long",
  "outcome": "win",
  "pnl_pct": 3.2,
  "description": "Price bounced from MA50 with RSI oversold"
}
```

### Полный паттерн

```json
{
  "id": "ma50_bounce_001",
  "name": "MA50 Bounce",
  "category": "trend_continuation",
  "direction": "long",
  
  "entry": {
    "description": "Price touched MA50 from above with RSI < 35",
    "indicators": {
      "rsi": 32,
      "price_to_ma50_pct": -0.5,
      "volume_ratio": 1.2
    },
    "timestamp": "2024-12-15T08:00:00Z",
    "price": 42500
  },
  
  "exit": {
    "reason": "take_profit",
    "timestamp": "2024-12-16T04:00:00Z",
    "price": 43860
  },
  
  "outcome": "win",
  "pnl_pct": 3.2,
  "holding_period_hours": 20,
  
  "context": {
    "symbol": "BTC_USDT",
    "timeframe": "4h",
    "regime": "normal",
    "trend": "bullish"
  },
  
  "lessons": [
    "MA50 bounce works well in bullish trend",
    "RSI confirmation improves win rate"
  ],
  
  "tags": ["ma50", "bounce", "oversold", "trend_continuation"],
  
  "metadata": {
    "source": "manual_analysis",
    "analyst": "ANT",
    "created_at": "2024-12-16T10:00:00Z",
    "version": 1
  }
}
```

---

## Field Reference

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Уникальный идентификатор (snake_case) |
| `name` | string | Человекочитаемое название |
| `direction` | enum | `"long"` / `"short"` / `"neutral"` |
| `outcome` | enum | `"win"` / `"loss"` / `"breakeven"` |
| `description` | string | Текстовое описание паттерна (для embeddings) |

### Recommended Fields

| Field | Type | Description |
|-------|------|-------------|
| `pnl_pct` | float | Результат в процентах (+3.2 = прибыль 3.2%) |
| `category` | string | Категория паттерна |
| `entry.indicators` | object | Значения индикаторов на входе |
| `context.symbol` | string | Торговая пара |
| `context.timeframe` | string | Таймфрейм (`"15m"`, `"1h"`, `"4h"`, `"1d"`) |
| `tags` | array | Теги для поиска |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `entry.timestamp` | ISO8601 | Время входа |
| `entry.price` | float | Цена входа |
| `exit.timestamp` | ISO8601 | Время выхода |
| `exit.price` | float | Цена выхода |
| `exit.reason` | enum | Причина выхода |
| `holding_period_hours` | int | Время в позиции |
| `context.regime` | enum | Режим рынка |
| `lessons` | array | Уроки из сделки |
| `metadata` | object | Дополнительные данные |

---

## Categories

Рекомендуемые категории паттернов:

| Category | Description |
|----------|-------------|
| `trend_continuation` | Продолжение тренда |
| `trend_reversal` | Разворот тренда |
| `momentum_breakout` | Пробой с импульсом |
| `false_breakout` | Ложный пробой |
| `range_trade` | Торговля в диапазоне |
| `support_bounce` | Отскок от поддержки |
| `resistance_rejection` | Отбой от сопротивления |
| `divergence` | Дивергенция |
| `consolidation_break` | Выход из консолидации |

---

## Exit Reasons

| Reason | Description |
|--------|-------------|
| `take_profit` | Достигнут тейк-профит |
| `stop_loss` | Сработал стоп-лосс |
| `trailing_stop` | Сработал трейлинг-стоп |
| `time_exit` | Выход по времени |
| `manual` | Ручной выход |
| `signal_reversal` | Сигнал развернулся |
| `risk_limit` | Достигнут лимит риска |

---

## Regime Types

| Regime | Description |
|--------|-------------|
| `normal` | Нормальная волатильность |
| `low_vol` | Низкая волатильность |
| `high_vol` | Высокая волатильность |
| `trending` | Направленный тренд |
| `ranging` | Боковое движение |
| `crisis` | Кризис / чёрный лебедь |

---

## File Formats

### Single Pattern (JSON)

```bash
rag/patterns/ma50_bounce_001.json
```

### Batch Import (JSON Array)

```json
[
  {
    "id": "pattern_001",
    "name": "Pattern 1",
    ...
  },
  {
    "id": "pattern_002",
    "name": "Pattern 2",
    ...
  }
]
```

### CSV Format

```csv
id,name,direction,outcome,pnl_pct,description,category
ma50_bounce_001,MA50 Bounce,long,win,3.2,"Price bounced from MA50",trend_continuation
rsi_oversold_001,RSI Oversold,long,win,2.1,"RSI below 30 reversal",trend_reversal
```

---

## Validation

Перед загрузкой паттерны валидируются:

```bash
python scripts/validate_patterns.py --file patterns.json
```

### Validation Rules

1. **Required fields** — все обязательные поля присутствуют
2. **ID uniqueness** — id уникален в системе
3. **Enum values** — direction, outcome, regime из списка
4. **Types** — pnl_pct это число, tags это массив
5. **Description length** — минимум 10 символов

### Validation Output

```
Validating patterns.json...

✓ pattern_001: valid
✓ pattern_002: valid
✗ pattern_003: missing 'outcome' field
✗ pattern_004: invalid direction 'up' (must be long/short/neutral)

Summary: 2 valid, 2 invalid
```

---

## Loading Patterns

### Single File

```bash
python scripts/load_patterns.py --file my_patterns.json
```

### Directory

```bash
python scripts/load_patterns.py --dir my_patterns/
```

### With Validation

```bash
python scripts/load_patterns.py --file my_patterns.json --validate
```

### Rebuild Index

```bash
python scripts/load_patterns.py --rebuild-index
```

---

## Best Practices

### 1. Descriptive Names

```json
// ✗ Bad
{ "name": "Pattern 1" }

// ✓ Good
{ "name": "MA50 Bounce with RSI Confirmation" }
```

### 2. Rich Descriptions

```json
// ✗ Bad
{ "description": "Long trade" }

// ✓ Good
{ "description": "Price touched MA50 from above after 3 red candles. RSI at 32 showed oversold. Volume 1.2x average confirmed buyer interest. Entered at $42,500 with target at previous high." }
```

### 3. Include Lessons

```json
"lessons": [
  "Wait for RSI confirmation before entry",
  "MA50 bounce has 70% win rate in bullish regime",
  "Avoid entries near key resistance"
]
```

### 4. Use Tags

```json
"tags": ["ma50", "bounce", "oversold", "trend_continuation", "btc", "4h"]
```

### 5. Context Matters

```json
"context": {
  "symbol": "BTC_USDT",
  "timeframe": "4h",
  "regime": "normal",
  "trend": "bullish",
  "btc_dominance": "high",
  "market_phase": "accumulation"
}
```

---

## Examples

### Trend Continuation

```json
{
  "id": "trend_cont_001",
  "name": "EMA21 Pullback Long",
  "category": "trend_continuation",
  "direction": "long",
  "outcome": "win",
  "pnl_pct": 4.5,
  "description": "In strong uptrend, price pulled back to EMA21 and bounced. MACD positive, volume increasing on bounce. Classic trend continuation setup.",
  "entry": {
    "indicators": {
      "rsi": 45,
      "price_to_ema21_pct": 0.2,
      "macd_histogram": 150,
      "volume_ratio": 1.3
    }
  },
  "context": {
    "symbol": "BTC_USDT",
    "timeframe": "4h",
    "regime": "trending",
    "trend": "bullish"
  },
  "tags": ["ema21", "pullback", "trend", "continuation"]
}
```

### Failed Breakout

```json
{
  "id": "false_break_001",
  "name": "Failed Resistance Break",
  "category": "false_breakout",
  "direction": "short",
  "outcome": "win",
  "pnl_pct": 2.8,
  "description": "Price broke above resistance on low volume, immediately rejected. RSI divergence warned of weakness. Shorted the rejection candle.",
  "entry": {
    "indicators": {
      "rsi": 72,
      "volume_ratio": 0.7,
      "price_above_resistance_pct": 0.5
    }
  },
  "lessons": [
    "Low volume breakouts often fail",
    "RSI divergence = warning sign",
    "Wait for rejection confirmation"
  ],
  "tags": ["false_breakout", "resistance", "divergence", "short"]
}
```

---

## Migration from Legacy Format

Если у вас есть паттерны в старом формате:

```bash
python scripts/migrate_patterns.py --from legacy/ --to new/
```

---

## Support

При проблемах с форматом:

1. Проверьте валидацию: `python scripts/validate_patterns.py`
2. Смотрите примеры в `rag/patterns/library.json`
3. Создайте issue в GitHub

---

*Last updated: 2024-12-20*