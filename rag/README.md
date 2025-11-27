# RAG Module - Инструкция для Егора 1

## Описание

Модуль RAG предназначен для извлечения и анализа торговых паттернов из криптовалютных данных.

## Структура

```
rag/
├── __init__.py         # Инициализация модуля
├── config.py           # Конфигурация (монеты, таймфреймы, пути)
├── extractor.py        # Основной класс для извлечения паттернов
├── cli.py              # Интерфейс командной строки
├── README.md           # Эта инструкция
├── requirements.txt    # Зависимости Python
└── patterns/           # Сохранённые паттерны (JSON)
    └── .gitkeep
```

## Установка

```bash
# Установить зависимости
pip install -r rag/requirements.txt
```

## Использование

### Командная строка (CLI)

```bash
# Извлечь бычьи паттерны для BTC на 15-минутном таймфрейме
python -m rag.cli --coin BTC --timeframe 15m --pattern-type bullish --stats

# Извлечь медвежьи паттерны для ETH на 1-часовом таймфрейме
python -m rag.cli --coin ETH --timeframe 1h --pattern-type bearish

# Использовать кастомный путь к данным
python -m rag.cli --coin SOL --timeframe 4h --data-path /path/to/data.csv
```

### Python API

```python
from rag import PatternExtractor

# Инициализация
extractor = PatternExtractor(coin='BTC', timeframe='15m')

# Загрузка данных
data = extractor.load_data()

# Извлечение паттернов
patterns = extractor.extract_patterns(pattern_type='bullish')

# Сохранение результатов
output_path = extractor.save_patterns()

# Получение статистики
stats = extractor.get_statistics()
print(stats)
```

## Параметры

### Поддерживаемые монеты
```
ALGO, AVAX, BTC, DOT, ENA, ETH, HBAR,
LDO, LINK, LTC, ONDO, SOL, SUI, UNI
```

### Поддерживаемые таймфреймы
```
15m, 1h, 4h, 1d
```

### Типы паттернов
- `bullish` - Бычьи паттерны (рост цены)
- `bearish` - Медвежьи паттерны (падение цены)
- `consolidation` - Консолидация (боковое движение)

## Конфигурация

Основные параметры в `config.py`:

```python
MIN_PATTERN_LENGTH = 10      # Минимальная длина паттерна (свечей)
MAX_PATTERN_LENGTH = 100     # Максимальная длина паттерна (свечей)
SIMILARITY_THRESHOLD = 0.85  # Порог схожести паттернов
FEATURE_WINDOW = 20          # Окно для расчёта признаков
```

## Формат выходных данных

Паттерны сохраняются в JSON:

```json
{
  "coin": "BTC",
  "timeframe": "15m",
  "patterns_count": 42,
  "extraction_date": "2025-11-27T20:40:00",
  "patterns": [
    {
      "type": "bullish",
      "start_idx": 100,
      "end_idx": 200,
      "start_time": "2025-11-01 10:00:00",
      "end_time": "2025-11-01 11:00:00",
      "length": 100,
      "return": 0.0235
    }
  ]
}
```

## Статистика паттернов

```python
{
  'total_patterns': 42,
  'avg_return': 0.0234,
  'median_return': 0.0198,
  'std_return': 0.0156,
  'min_return': -0.0045,
  'max_return': 0.0678
}
```

## Дальнейшая разработка

### Задачи для Егора 1:

1. **Улучшить детекцию паттернов** в `extractor.py`:
   - Добавить распознавание классических паттернов (Head & Shoulders, Double Top/Bottom)
   - Реализовать DTW (Dynamic Time Warping) для поиска похожих паттернов

2. **Добавить feature engineering**:
   - Интеграция с `core/feature_loader.py` для использования 75 признаков
   - Расчёт дополнительных индикаторов для паттернов

3. **Создать визуализацию**:
   - Построение графиков найденных паттернов
   - Интерактивный дашборд для анализа

4. **Оптимизация**:
   - Параллельная обработка нескольких монет/таймфреймов
   - Кэширование промежуточных результатов

## Интеграция с основным проектом

Модуль RAG интегрируется с:
- `data/raw/` - Исходные OHLCV данные
- `data/features/` - Признаки для ML моделей
- `strategies/` - Торговые стратегии используют найденные паттерны

## Вопросы?

Обращайтесь к STAR_ANT или создавайте issue в репозитории.
