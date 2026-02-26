"""
───────────────────────────────────────────────────────────────────────────────
REGIME ANNOTATION TOOL  —  Interactive CLI
───────────────────────────────────────────────────────────────────────────────

Lets you label regime segments from a trading session interactively and
appends them to data/regime_annotations.csv for use in supervised HMM training.

Usage:
    python annotate_regimes.py

Input format (flexible):
    09:15-10:30 Trending
    11:00 - 13:45 Choppy high
    14:00-15:25 Range medium
    10:10-10:50 choppy        ← case-insensitive
    (blank line or 'done' to finish the session)

Regimes:   Trending | Range | Choppy
Confidence: high (default) | medium | low

Author: Jeevan Jonas / Claude 2026
───────────────────────────────────────────────────────────────────────────────
"""

import csv
import os
import re
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_FILE = os.path.join(BASE_DIR, "data", "regime_annotations.csv")
VALID_REGIMES   = {"trending", "range", "choppy"}
VALID_CONF      = {"high", "medium", "low"}
DEFAULT_CONF    = "high"

# Direction-aware aliases — Trending-Up/Down both normalise to "Trending"
# The trainer only uses Trending/Range/Choppy; direction is inferred at inference.
REGIME_ALIASES = {
    "trending-up":   "Trending",  "trending-down": "Trending",
    "trending_up":   "Trending",  "trending_down": "Trending",
    "t-up":          "Trending",  "t-down":        "Trending",
    "tup":           "Trending",  "tdown":         "Trending",
    "tu":            "Trending",  "td":            "Trending",
    "t":             "Trending",
    "r":             "Range",
    "c":             "Choppy",
}

FIELDNAMES = ["datetime_start", "datetime_end", "regime", "confidence"]

# ── Colours (optional, graceful fallback) ────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    C_PROMPT  = Fore.CYAN
    C_OK      = Fore.GREEN
    C_WARN    = Fore.YELLOW
    C_ERR     = Fore.RED
    C_DIM     = Style.DIM
    C_RESET   = Style.RESET_ALL
except ImportError:
    C_PROMPT = C_OK = C_WARN = C_ERR = C_DIM = C_RESET = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header():
    print(f"\n{C_PROMPT}{'─'*60}")
    print("  REGIME ANNOTATION TOOL")
    print(f"{'─'*60}{C_RESET}")
    print(f"  Output: {ANNOTATION_FILE}")
    print(f"  Regimes: Trending[-Up/-Down] | T-Up | T-Down | Range | Choppy")
    print(f"  Confidence: high (default) | medium | low")
    print(f"\n  Format: HH:MM-HH:MM <regime> [confidence]")
    print(f"  Example: 09:15-10:30 Trending-Up  or just  Trending")
    print(f"           11:00-13:45 Choppy medium")
    print(f"           14:00-15:25 T-Down  (shorthand for Trending-Down)")
    print(f"           14:00-15:25 Range")
    print(f"\n  Commands: done / skip / quit / show\n")


def parse_date_input(raw: str) -> date | None:
    """Accept YYYY-MM-DD or DD-MM-YYYY or DD/MM/YYYY or 'today'."""
    raw = raw.strip().lower()
    if raw in ("today", "t"):
        return date.today()
    if raw in ("yesterday", "y"):
        return date.today() - timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_segment(line: str, session_date: date) -> dict | None:
    """
    Parse one annotation line into a dict ready for CSV.

    Accepted patterns:
        HH:MM-HH:MM regime [confidence]
        HH:MM - HH:MM regime [confidence]
    """
    line = line.strip()
    if not line:
        return None

    # Extract time range — allow spaces around dash
    time_pattern = r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})"
    m = re.search(time_pattern, line)
    if not m:
        return None

    t_start_str, t_end_str = m.group(1), m.group(2)
    remainder = line[m.end():].strip()

    # Parse regime and optional confidence from remainder
    tokens = remainder.lower().split()
    if not tokens:
        return None

    raw_regime = tokens[0]
    # Resolve alias first, then check against VALID_REGIMES
    regime = REGIME_ALIASES.get(raw_regime, raw_regime.capitalize())
    if regime.lower() not in VALID_REGIMES:
        return None

    confidence = DEFAULT_CONF
    if len(tokens) >= 2 and tokens[1] in VALID_CONF:
        confidence = tokens[1]

    # Build full datetimes
    def make_dt(t_str: str) -> datetime:
        h, mi = map(int, t_str.split(":"))
        return datetime.combine(session_date, datetime.min.time().replace(hour=h, minute=mi))

    try:
        dt_start = make_dt(t_start_str)
        dt_end   = make_dt(t_end_str)
    except ValueError:
        return None

    if dt_end <= dt_start:
        return None

    return {
        "datetime_start": dt_start.strftime("%Y-%m-%d %H:%M:%S"),
        "datetime_end":   dt_end.strftime("%Y-%m-%d %H:%M:%S"),
        "regime":         regime,
        "confidence":     confidence,
    }


def load_existing() -> list[dict]:
    """Load existing annotations from CSV, return list of dicts."""
    if not os.path.exists(ANNOTATION_FILE):
        return []
    with open(ANNOTATION_FILE, newline="") as f:
        return list(csv.DictReader(f))


def save_annotations(rows: list[dict]):
    """Write all rows to CSV, sorted by datetime_start."""
    rows_sorted = sorted(rows, key=lambda r: r["datetime_start"])
    os.makedirs(os.path.dirname(ANNOTATION_FILE), exist_ok=True)
    with open(ANNOTATION_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_sorted)


def show_annotations(rows: list[dict], session_date: date | None = None):
    """Print current annotations, optionally filtered to one date."""
    if not rows:
        print(f"  {C_DIM}(no annotations yet){C_RESET}")
        return
    date_str = session_date.strftime("%Y-%m-%d") if session_date else None
    shown = [r for r in rows if (date_str is None or r["datetime_start"].startswith(date_str))]
    if not shown:
        print(f"  {C_DIM}(no annotations for {date_str}){C_RESET}")
        return
    print(f"\n  {'START':<22} {'END':<22} {'REGIME':<12} {'CONF'}")
    print(f"  {'─'*22} {'─'*22} {'─'*12} {'─'*8}")
    for r in shown:
        regime_col = {
            "Trending": C_OK,
            "Range":    C_PROMPT,
            "Choppy":   C_WARN,
        }.get(r["regime"], "")
        print(f"  {r['datetime_start']:<22} {r['datetime_end']:<22} "
              f"{regime_col}{r['regime']:<12}{C_RESET} {r['confidence']}")
    print()


def confirm(prompt: str) -> bool:
    ans = input(f"{C_PROMPT}{prompt} [y/N]: {C_RESET}").strip().lower()
    return ans in ("y", "yes")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    print_header()

    # Load existing
    all_rows = load_existing()
    print(f"  {C_DIM}Loaded {len(all_rows)} existing annotation(s).{C_RESET}\n")

    while True:
        # ── Ask for date ──────────────────────────────────────────────────────
        raw_date = input(
            f"{C_PROMPT}Session date{C_RESET} (YYYY-MM-DD, 'today', 'yesterday', or 'quit'): "
        ).strip()

        if raw_date.lower() in ("quit", "q", "exit"):
            print(f"\n{C_OK}Saved {len(all_rows)} annotation(s) to:{C_RESET}")
            print(f"  {ANNOTATION_FILE}")
            break

        if raw_date.lower() == "show":
            show_annotations(all_rows)
            continue

        session_date = parse_date_input(raw_date)
        if session_date is None:
            print(f"  {C_ERR}Could not parse date. Try: 2022-06-30 or 30-06-2022{C_RESET}\n")
            continue

        date_label = session_date.strftime("%Y-%m-%d (%A)")
        print(f"\n  {C_OK}Session: {date_label}{C_RESET}")
        print(f"  Enter segments one per line. Blank line or 'done' to finish.\n")

        # Show existing annotations for this date
        show_annotations(all_rows, session_date)

        session_rows = []

        while True:
            try:
                raw = input(f"  {C_PROMPT}>{C_RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not raw:
                break

            cmd = raw.lower()

            if cmd in ("done", "d"):
                break

            if cmd in ("quit", "q", "exit"):
                # Save and exit entirely
                if session_rows:
                    all_rows.extend(session_rows)
                    save_annotations(all_rows)
                print(f"\n{C_OK}Saved. Exiting.{C_RESET}")
                sys.exit(0)

            if cmd == "show":
                show_annotations(all_rows, session_date)
                continue

            if cmd == "skip":
                print(f"  {C_DIM}Session skipped.{C_RESET}")
                break

            if cmd in ("undo", "u") and session_rows:
                removed = session_rows.pop()
                print(f"  {C_WARN}Removed: {removed['datetime_start']} – "
                      f"{removed['datetime_end']} {removed['regime']}{C_RESET}")
                continue

            if cmd.startswith("help") or cmd == "?":
                print(f"  Format:  HH:MM-HH:MM <Regime> [confidence]  (direction optional)")
                print(f"  Example: 09:15-10:30 Trending")
                print(f"           11:00-13:45 Choppy medium")
                print(f"  Commands: done | skip | show | undo | quit")
                continue

            seg = parse_segment(raw, session_date)
            if seg is None:
                print(f"  {C_ERR}Could not parse: '{raw}'{C_RESET}")
                print(f"  {C_DIM}Expected: HH:MM-HH:MM Regime [confidence]{C_RESET}")
                continue

            session_rows.append(seg)
            print(f"  {C_OK}✔  {seg['datetime_start']} → {seg['datetime_end']}  "
                  f"{seg['regime']} ({seg['confidence']}){C_RESET}")

        # ── Confirm and save session ──────────────────────────────────────────
        if session_rows:
            print(f"\n  {C_PROMPT}Summary for {date_label}:{C_RESET}")
            for s in session_rows:
                print(f"    {s['datetime_start'][11:16]}–{s['datetime_end'][11:16]}  "
                      f"{s['regime']:<12} {s['confidence']}")

            if confirm("\n  Save these annotations?"):
                # Remove any existing annotations for this date before appending
                date_str = session_date.strftime("%Y-%m-%d")
                existing_other = [r for r in all_rows
                                  if not r["datetime_start"].startswith(date_str)]
                existing_same  = [r for r in all_rows
                                  if r["datetime_start"].startswith(date_str)]

                if existing_same:
                    print(f"  {C_WARN}Found {len(existing_same)} existing annotation(s) "
                          f"for {date_str}.{C_RESET}")
                    if confirm("  Replace them with the new ones?"):
                        all_rows = existing_other + session_rows
                    else:
                        all_rows = all_rows + session_rows
                else:
                    all_rows.extend(session_rows)

                save_annotations(all_rows)
                print(f"  {C_OK}✔  Saved. Total annotations: {len(all_rows)}{C_RESET}\n")
            else:
                print(f"  {C_WARN}Discarded.{C_RESET}\n")
        else:
            print(f"  {C_DIM}No segments entered for {date_label}.{C_RESET}\n")

        # Ask to continue with another date
        if not confirm("  Annotate another session?"):
            print(f"\n{C_OK}Done. {len(all_rows)} annotation(s) in:{C_RESET}")
            print(f"  {ANNOTATION_FILE}\n")
            break


if __name__ == "__main__":
    run()