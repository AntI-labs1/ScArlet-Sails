# ScArlet-Sails on Kaggle

Запуск smoke-тестов и vectorbt-движка в Kaggle-нотбуке.

## Шаги

1. **Запушить ветку с ревизией** в GitHub (origin = StarDust1508/ScArlet-Sails).
   Текущая ветка ревизии: `claude/quizzical-raman-434cfb`.
   ```bash
   git push -u origin claude/quizzical-raman-434cfb
   ```

2. **Создать новый Kaggle Notebook**:
   - https://www.kaggle.com/code → New Notebook
   - File → Import Notebook → выбрать `kaggle/smoke_test.ipynb`
   - Settings → Accelerator: **None** (CPU достаточно)
   - Settings → Internet: **On** (нужно для `git clone` и `pip install`)

3. **Запустить все ячейки** (Run All). Время ~7-10 минут (большая часть — pip install vectorbt).

## Что проверяет нотбук

| Шаг | Что проверяет | Ожидание |
|---|---|---|
| 1. Clone | репо доступен, ветка корректная | без ошибок |
| 2. Install | requirements.txt валиден, deps совместимы | `pip install` без conflict-логов |
| 3. pytest | новые fixtures работают, исправленные баги не регрессируют | passed >> failed |
| 4. VBT smoke (single) | новый движок живой, метрики считаются | `num_trades > 0`, не ImportError |
| 5. VBT smoke (multi) | multi-asset работает | DataFrame с метриками по 4 монетам |

## Если что-то падает

- **`git clone` 403**: ветка ещё не запушена в GitHub. Запуши и повтори.
- **`pip install` тормозит/таймит**: Kaggle Internet выключен. Включи в настройках.
- **`pytest` падает на data_loader**: проверь, что `tests/conftest.py` отработал — он должен создать
  синтетические parquet в `data/raw/`. Если папка пустая — глянь права на запись.
- **`VBT` падает с ImportError**: vectorbt не доустановился (numba/llvmlite могут конфликтовать с
  пред-установленным Kaggle numpy). Попробуй `!pip install -q --upgrade vectorbt`.
- **`Portfolio.from_signals` ругается на параметры**: возможна несовместимость с конкретной
  версией vectorbt — пришли traceback, докручу `backtesting/vbt_engine.py`.

## Альтернатива: запуск из чистой Python без Kaggle

На любой машине с Python 3.10+:
```bash
git clone -b claude/quizzical-raman-434cfb https://github.com/StarDust1508/ScArlet-Sails.git
cd ScArlet-Sails
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
python run_backtest.py --strategy rsi --coin BTC --timeframe 15m
```
Синтетические фикстуры подтянутся автоматически.
