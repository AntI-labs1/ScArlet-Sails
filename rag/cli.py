#!/usr/bin/env python3
"""
Scarlet Sails RAG CLI
=====================

Командная строка для извлечения данных паттернов.

Использование:
    python -m rag.cli --coin BTC --tf 1h --time "2024-11-26 14:00"
    
Сокращённая форма:
    python -m rag.cli BTC 1h "2024-11-26 14:00"
"""

import argparse
import sys
import json
import random
from pathlib import Path

import pandas as pd

from core.feature_engine_v2 import CanonicalMarketStateBuilder
from .extractor import PatternExtractor
from .config import COINS, TIMEFRAMES, PATTERNS_DIR, TimeCapsuleSnapshot


def print_banner():
    """Красивый баннер."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         SCARLET SAILS — RAG PATTERN EXTRACTOR             ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_result(data: dict):
    """Красивый вывод результата."""
    if "error" in data:
        print(f"\n❌ ОШИБКА: {data['error']}")
        return
    
    print("\n" + "="*60)
    print(f"📊 ПАТТЕРН: {data['id']}")
    print("="*60)
    
    meta = data.get('meta', {})
    print(f"\n🪙  Монета:     {meta.get('coin')}")
    print(f"⏰  Таймфрейм:  {meta.get('timeframe')}")
    print(f"📈  Тип:        {meta.get('pattern_type')}")
    print(f"↗️   Направление: {meta.get('direction')}")
    
    timing = data.get('timing', {})
    print(f"\n🕐  Время пробития: {timing.get('breakout_time_actual')}")
    print(f"🕐  Время setup:    {timing.get('setup_time')}")
    
    box = data.get('box', {})
    if box and "error" not in box:
        print(f"\n📦 BOX METRICS:")
        print(f"   Support:     {box.get('support')}")
        print(f"   Resistance:  {box.get('resistance')}")
        print(f"   Range:       {box.get('box_range_pct')}%")
        print(f"   Touches S:   {box.get('touches_support')}")
        print(f"   Touches R:   {box.get('touches_resistance')}")
        print(f"   Duration:    {box.get('duration_bars')} bars")
    
    ind = data.get('indicators_before', {})
    print(f"\n📉 ИНДИКАТОРЫ (до пробития):")
    print(f"   RSI z-score:     {ind.get('rsi_zscore')}")
    print(f"   MACD z-score:    {ind.get('macd_zscore')}")
    print(f"   ATR z-score:     {ind.get('atr_zscore')}")
    print(f"   Volume z-score:  {ind.get('volume_zscore')}")
    print(f"   Trend Up:        {ind.get('trend_up')}")
    print(f"   Vol Low:         {ind.get('vol_low')}")
    
    w = data.get('w_box', {})
    if w:
        print(f"\n🎯 W_BOX КОМПОНЕНТЫ:")
        print(f"   I_rsi:        {w.get('I_rsi')}")
        print(f"   I_volatility: {w.get('I_volatility')}")
        print(f"   I_volume:     {w.get('I_volume')}")
        print(f"   I_touches:    {w.get('I_touches')}")
        print(f"   ────────────────────")
        print(f"   W_BOX:        {w.get('W_box')} {'✅' if w.get('W_box', 0) > 0.3 else '⚠️'}")
    
    print("\n" + "="*60)


def cmd_extract(args):
    """Команда извлечения паттерна."""
    print(f"\n🔍 Поиск: {args.coin} {args.tf} @ {args.time}...")
    
    try:
        extractor = PatternExtractor(args.coin, args.tf)
        data = extractor.extract(
            breakout_time=args.time,
            pattern_type=args.type,
            direction=args.direction,
            lookback=args.lookback,
            notes=args.notes or ""
        )
        
        print_result(data)
        
        if "error" not in data:
            path = extractor.save(data)
            if path:
                print(f"\n💾 Файл: {path}")
                print(f"\n📤 Для отправки в GitHub:")
                print(f"   git add {path}")
                pattern_id = data["id"]
                print(f"   git commit -m 'Pattern: {pattern_id}'")
                print(f"   git push")
        
    except FileNotFoundError as e:
        print(f"\n❌ Файл данных не найден: {e}")
        print("   Выполни: git pull")
    except Exception as e:
        print(f"\n💥 Ошибка: {e}")
        sys.exit(1)


def cmd_list(args):
    """Команда списка паттернов."""
    patterns = list(PATTERNS_DIR.glob("*.json"))
    
    if not patterns:
        print("\n📭 Паттернов пока нет.")
        print(f"   Папка: {PATTERNS_DIR}")
        return
    
    print(f"\n📋 ПАТТЕРНЫ ({len(patterns)}):")
    print("-"*60)
    
    for p in sorted(patterns):
        with open(p, 'r') as f:
            data = json.load(f)
        
        meta = data.get('meta', {})
        w = data.get('w_box', {}).get('W_box', '?')
        print(f"   {p.stem}")
        print(f"      {meta.get('coin')} {meta.get('timeframe')} | W_box: {w}")
    
    print("-"*60)


def cmd_stats(args):
    """Команда статистики."""
    patterns = list(PATTERNS_DIR.glob("*.json"))
    
    if not patterns:
        print("\n📭 Паттернов пока нет.")
        return
    
    coins = {}
    timeframes = {}
    w_box_values = []
    
    for p in patterns:
        with open(p, 'r') as f:
            data = json.load(f)
        
        meta = data.get('meta', {})
        coin = meta.get('coin', '?')
        tf = meta.get('timeframe', '?')
        w = data.get('w_box', {}).get('W_box')
        
        coins[coin] = coins.get(coin, 0) + 1
        timeframes[tf] = timeframes.get(tf, 0) + 1
        if w is not None:
            w_box_values.append(w)
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Всего паттернов: {len(patterns)}")
    
    print(f"\n   По монетам:")
    for c, n in sorted(coins.items(), key=lambda x: -x[1]):
        print(f"      {c}: {n}")
    
    print(f"\n   По таймфреймам:")
    for t, n in sorted(timeframes.items()):
        print(f"      {t}: {n}")
    
    if w_box_values:
        avg_w = sum(w_box_values) / len(w_box_values)
        good = sum(1 for w in w_box_values if w > 0.3)
        print(f"\n   W_box:")
        print(f"      Средний: {avg_w:.4f}")
        print(f"      Хороших (>0.3): {good} ({100*good/len(w_box_values):.0f}%)")


def cmd_record_pattern(args):
    """Create a Time Capsule snapshot from a CSV/Parquet bar index."""
    df = pd.read_csv(args.path) if args.path.endswith(".csv") else pd.read_parquet(args.path)
    if args.timestamp_column and args.timestamp_column in df.columns:
        df[args.timestamp_column] = pd.to_datetime(df[args.timestamp_column])
        df.set_index(args.timestamp_column, inplace=True)

    builder = CanonicalMarketStateBuilder(df)
    market_state = builder.build_for_index(args.bar_index)

    snapshot = TimeCapsuleSnapshot(
        timestamp=str(df.index[args.bar_index]),
        symbol=args.coin,
        timeframe=args.tf,
        market_state_window=market_state.get("window_slice", {}).to_dict() if hasattr(market_state.get("window_slice", {}), "to_dict") else {},
        P_rb=args.P_rb,
        P_ml=args.P_ml,
        P_hyb=args.P_hyb,
        regime=market_state.get("regime", "unknown"),
        pattern_type=args.pattern_type,
        human_label=args.label,
        human_confidence=args.confidence,
        reviewed_by=None,
        trade_pnl=None,
        metadata={"notes": args.notes} if args.notes else None,
    )

    out_path = PATTERNS_DIR / f"snapshot_{args.coin}_{args.tf}_{args.bar_index}.json"
    with open(out_path, "w") as f:
        json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"Saved snapshot → {out_path}")


def cmd_sample_for_audit(args):
    """Sample Time Capsule patterns for quick human audit."""
    patterns = list(PATTERNS_DIR.glob("*.json"))
    if not patterns:
        print("Нет паттернов для аудита.")
        return

    sample = random.sample(patterns, min(args.n, len(patterns)))
    results = []
    for path in sample:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"\nID: {path.stem} | ts={data.get('timestamp', '?')} | regime={data.get('regime', '?')} | P_rb={data.get('P_rb')}")
        decision = input("Mark [ok/questionable/bad/skip]: ").strip() or "skip"
        data["audit_tag"] = decision
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        results.append(decision)

    print(f"\nAudit completed. Tags: {dict(pd.Series(results).value_counts())}")


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Scarlet Sails RAG Pattern Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m rag.cli --coin BTC --tf 1h --time "2024-11-26 14:00"
  python -m rag.cli BTC 15m "2024-11-26 14:30" --direction short
  python -m rag.cli --list
  python -m rag.cli --stats
        """
    )
    
    subparsers = parser.add_subparsers(dest="command")

    # Default extract command (backward compatible positional usage)
    parser_extract = subparsers.add_parser("extract", add_help=False)
    parser_extract.add_argument('coin', nargs='?', type=str, help='Монета (BTC, ETH, ...)')
    parser_extract.add_argument('tf', nargs='?', type=str, choices=TIMEFRAMES, help='Таймфрейм')
    parser_extract.add_argument('time', nargs='?', type=str, help='Время "YYYY-MM-DD HH:MM"')
    parser_extract.add_argument('--coin', dest='coin_named', type=str, help='Монета')
    parser_extract.add_argument('--tf', dest='tf_named', type=str, choices=TIMEFRAMES, help='Таймфрейм')
    parser_extract.add_argument('--time', dest='time_named', type=str, help='Время')
    parser_extract.add_argument('--type', default='box_range', help='Тип паттерна (по умолчанию box_range)')
    parser_extract.add_argument('--direction', '-d', default='long', choices=['long', 'short'], help='Направление')
    parser_extract.add_argument('--lookback', '-l', type=int, default=48, help='Баров назад для box (по умолчанию 48)')
    parser_extract.add_argument('--notes', '-n', type=str, help='Заметки')
    parser_extract.add_argument('--list', action='store_true', help='Показать все паттерны')
    parser_extract.add_argument('--stats', action='store_true', help='Показать статистику')

    # Record pattern command
    parser_record = subparsers.add_parser("record-pattern")
    parser_record.add_argument('--path', required=True, help='CSV/Parquet source with features')
    parser_record.add_argument('--bar-index', type=int, required=True, help='Index of the bar to snapshot')
    parser_record.add_argument('--coin', required=True, help='Symbol')
    parser_record.add_argument('--tf', required=True, choices=TIMEFRAMES)
    parser_record.add_argument('--timestamp-column', default='timestamp', help='Timestamp column name')
    parser_record.add_argument('--pattern-type', default='manual', help='Pattern type label')
    parser_record.add_argument('--label', help='Human label')
    parser_record.add_argument('--confidence', type=float, help='Human confidence (0-1)')
    parser_record.add_argument('--notes', help='Notes to store')
    parser_record.add_argument('--P-rb', dest='P_rb', type=float, default=None, help='Optional P_rb score')
    parser_record.add_argument('--P-ml', dest='P_ml', type=float, default=None, help='Optional P_ml score')
    parser_record.add_argument('--P-hyb', dest='P_hyb', type=float, default=None, help='Optional P_hyb score')

    # Audit sampler
    parser_audit = subparsers.add_parser("sample-for-audit")
    parser_audit.add_argument('--n', type=int, default=20, help='Number of samples to review')

    argv = sys.argv[1:]
    if argv and argv[0] not in {"extract", "record-pattern", "sample-for-audit", "--help", "-h"} and not argv[0].startswith('-'):
        argv = ["extract"] + argv

    args = parser.parse_args(argv)
    
    print_banner()
    
    # Обработка команд
    if args.command == "record-pattern":
        cmd_record_pattern(args)
        return

    if args.command == "sample-for-audit":
        cmd_sample_for_audit(args)
        return

    # Default: extract flow
    if args.command is None:
        # mimic old positional usage
        args.command = "extract"
    if args.command == "extract":
        if args.list:
            cmd_list(args)
            return

        if args.stats:
            cmd_stats(args)
            return

        # Объединить позиционные и именованные
        coin = args.coin_named or getattr(args, 'coin', None)
        tf = args.tf_named or getattr(args, 'tf', None)
        time = args.time_named or getattr(args, 'time', None)

        if not all([coin, tf, time]):
            parser.print_help()
            print("\n❌ Нужно указать: монету, таймфрейм и время")
            print("\nПример:")
            print('   python -m rag.cli BTC 1h "2024-11-26 14:00"')
            sys.exit(1)

        # Валидация
        if coin.upper() not in COINS:
            print(f"\n❌ Монета {coin} не поддерживается.")
            print(f"   Доступные: {', '.join(COINS)}")
            sys.exit(1)

        # Установить значения
        args.coin = coin.upper()
        args.tf = tf
        args.time = time

        cmd_extract(args)


if __name__ == "__main__":
    main()