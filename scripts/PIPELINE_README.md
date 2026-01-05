# 🛡️ Canonical Data Pipeline v3.0

## Философия: "Железный Купол"

**ТОЛЬКО проверенные данные проходят дальше.**

Canonical Data Pipeline - это система валидации данных, которая гарантирует, что в систему попадают только корректные данные. Философия "Железного Купола" означает строгую многоуровневую защиту от невалидных данных.

## 📐 Архитектура

```
Raw Data → Validation Layer → Canonical Format → System
                ↓
             REJECT ❌
```

### Принципы:

1. **Single Source of Truth** - один канонический формат для всех данных
2. **Fail-Fast Principle** - отклонять данные при первой ошибке
3. **Структурная валидация** - проверка всех обязательных полей
4. **Временная согласованность** - данные должны быть актуальными

## 📦 Компоненты

### 1. `canonical_pipeline.py`

**Основной модуль валидации данных.**

#### Возможности:
- ✅ Структурная валидация (required fields)
- ✅ Типовая валидация (int, float, str)
- ✅ Валидация диапазонов (min/max price, volume)
- ✅ Временная валидация (data freshness)
- 📊 Статистика обработки
- 🔄 Пакетная обработка

#### Использование как библиотека:

```python
from scripts.canonical_pipeline import CanonicalPipeline, ValidationError

# Создаём pipeline
pipeline = CanonicalPipeline()

# Валидируем данные
raw_data = {
    "timestamp": "2025-01-18T12:00:00",
    "symbol": "BTC/USDT",
    "price": 45000.0,
    "volume": 1.5,
    "source": "binance"
}

try:
    canonical = pipeline.validate(raw_data)
    print(f"✅ Valid data: {canonical}")
except ValidationError as e:
    print(f"❌ Validation failed: {e}")

# Пакетная обработка
valid_data = pipeline.process_batch([data1, data2, data3])

# Статистика
stats = pipeline.get_stats()
print(f"Success rate: {stats['success_rate']:.2f}%")
```

### 2. `validate_data.py`

**CLI утилита для валидации данных.**

#### Команды:

```bash
# Проверить один файл
python scripts/validate_data.py data/market_data.json

# Проверить всю директорию
python scripts/validate_data.py --check-all

# Проверить данные от конкретного источника
python scripts/validate_data.py --source binance

# Проверить свежесть данных (последние 24 часа)
python scripts/validate_data.py --check-recent 24

# Сохранить отчёт
python scripts/validate_data.py data/ --output report.json

# Verbose режим
python scripts/validate_data.py data/ --verbose
```

#### Пример вывода:

```
📄 Validating: data/binance_btc.json
✅ Valid: 150/150 | ❌ Invalid: 0

============================================================
📊 VALIDATION REPORT
============================================================
{
  "file": "data/binance_btc.json",
  "total": 150,
  "valid": 150,
  "invalid": 0,
  "errors": []
}

============================================================
📊 PIPELINE STATISTICS
============================================================
  processed: 150
  valid: 150
  invalid: 0
  rejected: 0
  success_rate: 100.00
```

### 3. `cleanup_legacy.py`

**Протокол "Чистка" - утилита для удаления устаревших файлов.**

#### Команды:

```bash
# Dry-run (по умолчанию) - показать что будет удалено
python scripts/cleanup_legacy.py --dry-run

# Фактическое удаление (требует подтверждения)
python scripts/cleanup_legacy.py --execute

# Verbose режим
python scripts/cleanup_legacy.py --verbose
```

#### Что удаляется:

- `**/*_backup.py` - файлы с бэкапами
- `**/*_old.py` - старые версии
- `**/*_deprecated.py` - устаревшие файлы
- `**/__pycache__` - кэш Python
- `**/*.pyc` - скомпилированные файлы
- `**/.DS_Store` - системные файлы macOS

#### Безопасность:

1. **Dry-run по умолчанию** - всегда сначала просмотр
2. **Подтверждение** - требует ввода "YES" перед удалением
3. **Логирование** - все операции логируются

## 🔧 Конфигурация

### Создание custom конфига:

```json
{
  "min_price": 0.01,
  "max_price": 1000000,
  "min_volume": 0.001,
  "required_fields": ["timestamp", "symbol", "price", "volume"],
  "allowed_sources": ["binance", "coinbase", "kraken"],
  "max_age_seconds": 300
}
```

### Использование custom конфига:

```python
from pathlib import Path

pipeline = CanonicalPipeline(config_path=Path("config/validation.json"))
```

## 🧪 Тестирование

```bash
# Запустить тест pipeline
python scripts/test_pipeline.py
```

### Что тестируется:

- ✅ Инициализация pipeline
- ✅ Загрузка состояния
- ✅ Валидация features
- ✅ Обработка ошибок
- ✅ Pipeline успешно завершается

## 📊 Мониторинг

### Ключевые метрики:

```python
stats = pipeline.get_stats()

print(f"Processed: {stats['processed']}")
print(f"Valid: {stats['valid']}")
print(f"Invalid: {stats['invalid']}")
print(f"Rejected: {stats['rejected']}")
print(f"Success rate: {stats['success_rate']:.2f}%")
```

### Рекомендуемые alerts:

- ⚠️ Success rate < 95%
- ⚠️ No fresh data in last hour
- ⚠️ Rejection rate > 10%

## 🔒 Безопасность

### Валидация защищает от:

1. **Injection атак** - строгая типизация
2. **Данных из прошлого/будущего** - временная валидация
3. **Некорректных диапазонов** - min/max проверки
4. **Отсутствующих полей** - структурная валидация
5. **Неизвестных источников** - whitelist источников

## 🚀 Quick Start

### 1. Валидация в коде:

```python
from scripts.canonical_pipeline import CanonicalPipeline

pipeline = CanonicalPipeline()
valid_data = pipeline.validate(raw_data)
```

### 2. Валидация через CLI:

```bash
python scripts/validate_data.py data/
```

### 3. Чистка legacy файлов:

```bash
# Сначала dry-run
python scripts/cleanup_legacy.py

# Затем execute
python scripts/cleanup_legacy.py --execute
```

## 📚 Best Practices

### DO:

✅ Всегда валидировать данные перед использованием  
✅ Использовать canonical format во всей системе  
✅ Мониторить success rate  
✅ Логировать все ошибки валидации  
✅ Использовать dry-run перед cleanup  

### DON'T:

❌ Обходить валидацию  
❌ Использовать raw данные напрямую  
❌ Игнорировать ошибки валидации  
❌ Удалять файлы без dry-run  
❌ Создавать custom форматы данных  

## 🔄 Integration

### Интеграция в существующий код:

```python
# BEFORE
def process_market_data(raw_data):
    # Используем данные напрямую
    price = raw_data["price"]
    return analyze(price)

# AFTER
from scripts.canonical_pipeline import CanonicalPipeline, ValidationError

pipeline = CanonicalPipeline()

def process_market_data(raw_data):
    try:
        # Валидируем через pipeline
        canonical = pipeline.validate(raw_data)
        return analyze(canonical.price)
    except ValidationError as e:
        logger.error(f"Invalid data: {e}")
        return None
```

## 🐛 Troubleshooting

### Частые ошибки:

#### 1. "Missing required field: price"
```python
# Проверьте, что все обязательные поля присутствуют
required = ["timestamp", "symbol", "price", "volume"]
```

#### 2. "Price out of range"
```python
# Проверьте min_price и max_price в конфиге
config = {"min_price": 0.0001, "max_price": 1000000}
```

#### 3. "Data too old"
```python
# Увеличьте max_age_seconds в конфиге
config = {"max_age_seconds": 600}  # 10 minutes
```

## 📞 Support

При возникновении проблем:

1. Проверьте логи validation
2. Используйте `--verbose` режим
3. Проверьте конфигурацию
4. Запустите `test_pipeline.py`

## 🎯 Roadmap

- [ ] Async валидация
- [ ] Кэширование результатов
- [ ] Метрики в Prometheus
- [ ] GraphQL API для валидации
- [ ] Machine learning для anomaly detection

---

**Created:** 2025-01-18  
**Version:** 3.0  
**Status:** Production Ready ✅
