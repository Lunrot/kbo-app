#!/usr/bin/env python3
"""
KBO 2026 경기 결과 크롤러
https://sports.daum.net/prx/hermes/api/game/schedule.json 을 직접 호출하여
2026 시즌 페넌트레이스 경기 결과를 수집하고 2026.csv를 생성/업데이트합니다.

사용법:
  python3 crawl_2026.py

추가 패키지 설치 불필요 (표준 라이브러리만 사용)
"""

import os
import csv
import json
import calendar
import urllib.request
from datetime import datetime, date
from collections import defaultdict

# ─────────────────────────── 상수 ─────────────────────────────────────────────

# TEAMS = ['KIA', '삼성', 'LG', '두산', 'KT', 'SSG', '롯데', '한화', 'NC', '키움']
TEAMS = ['LG', '한화', 'SSG', '삼성', 'NC', 'KT', '롯데', 'KIA', '두산', '키움']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, '2026.csv')
YEAR = 2026
MONTHS = list(range(3, 11))  # 3월 ~ 10월 (시즌이 10월 초까지 이어질 수 있음)

API_URL = "https://sports.daum.net/prx/hermes/api/game/schedule.json"
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://sports.daum.net/schedule/kbo',
}

# ─────────────────────────── API 호출 ─────────────────────────────────────────

def fetch_month(year: int, month: int) -> dict:
    """특정 월의 KBO 경기 일정을 API에서 가져옵니다."""
    last_day = calendar.monthrange(year, month)[1]
    from_date = f"{year}{month:02d}01"
    to_date   = f"{year}{month:02d}{last_day:02d}"

    url = (
        f"{API_URL}?page=1&leagueCode=kbo&seasonKey={year}"
        f"&fromDate={from_date}&toDate={to_date}"
    )

    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def parse_pennant_games(data: dict, year: int, month: int, cutoff: date) -> list:
    """
    API 응답에서 페넌트레이스 완료 경기만 추출합니다.
    반환: list of dict
      {year, month, day, game_no, home_team, away_team, home_score, away_score}
    """
    games = []
    schedule = data.get('schedule', {})

    for date_str, day_games in schedule.items():
        y = int(date_str[:4])
        m = int(date_str[4:6])
        d = int(date_str[6:])

        if y != year or m != month:
            continue
        if date(y, m, d) > cutoff:
            continue  # 현재 시각 이후 경기 제외

        for game in day_games:
            if not isinstance(game, dict):
                continue

            # 페넌트레이스만 수집
            detail_type = (game.get('gameDetailType') or {}).get('nameKo', '')
            if detail_type != '페넌트레이스':
                continue

            # 결과가 없으면(미경기/취소) 제외
            home_result = game.get('homeResult')
            away_result = game.get('awayResult')
            if home_result is None or away_result is None:
                continue

            home_team = game.get('homeTeamName', '').strip()
            away_team = game.get('awayTeamName', '').strip()
            if home_team not in TEAMS or away_team not in TEAMS:
                continue

            games.append({
                'year': y, 'month': m, 'day': d,
                'game_id': int(game.get('gameId', 0)),
                'start_time': str(game.get('startTime', '0000')),
                'home_team': home_team,
                'away_team': away_team,
                'home_score': int(home_result),
                'away_score': int(away_result),
            })

    return games

# ─────────────────────────── 경기 → 컬럼/결과 변환 ───────────────────────────

def col_key(month: int, day: int, game_no: int = 1) -> str:
    """날짜 컬럼 키. 더블헤더는 M.D.N 형식."""
    return f"{month}.{day}" if game_no <= 1 else f"{month}.{day}.{game_no}"


def sort_col(col: str):
    """날짜 컬럼 정렬 키."""
    p = col.split('.')
    return (int(p[0]), int(p[1]), int(p[2]) if len(p) > 2 else 0)


def games_to_cols_results(games: list):
    """
    경기 목록 → (날짜컬럼 리스트, 팀별 결과 dict)

    더블헤더 판별: 하루에 2개의 다른 startTime이 있으면 더블헤더.
      - 빠른 시간대 경기 → M.D.1
      - 늦은 시간대 경기 → M.D.2
    단일 시간이지만 같은 팀 쌍이 2경기인 경우: gameId 순으로 .1/.2 분리.
    """
    day_map = defaultdict(list)
    for g in games:
        day_map[(g['month'], g['day'])].append(g)

    date_cols    = []
    team_results = {t: {} for t in TEAMS}

    for (mo, d) in sorted(day_map.keys()):
        day_games = day_map[(mo, d)]

        # 더블헤더 판별: 같은 팀이 하루에 2경기 이상 출전하는 경우만
        is_dh = any(
            sum(1 for g in day_games
                if g['home_team'] == t or g['away_team'] == t) >= 2
            for t in TEAMS
        )
        # 고유 시작 시간 목록 (오름차순)
        times = sorted(set(g['start_time'] for g in day_games))

        if is_dh:
            date_cols += [f"{mo}.{d}.1", f"{mo}.{d}.2"]

            if len(times) >= 2:
                # startTime 기준 분리: 가장 이른 시간 = .1, 나머지 = .2
                t1, t2 = times[0], times[-1]
                g1 = [g for g in day_games if g['start_time'] == t1]
                g2 = [g for g in day_games if g['start_time'] != t1]
            else:
                # 시간이 같으면 gameId 순으로 각 팀 쌍의 첫 경기 = .1, 두 번째 = .2
                pairs = defaultdict(list)
                for g in day_games:
                    key = tuple(sorted([g['home_team'], g['away_team']]))
                    pairs[key].append(g)
                g1, g2 = [], []
                for pl in pairs.values():
                    pl.sort(key=lambda g: g['game_id'])
                    g1.append(pl[0])
                    if len(pl) >= 2:
                        g2.append(pl[1])

            for slot, slot_games in [(1, g1), (2, g2)]:
                col = f"{mo}.{d}.{slot}"
                played = set()
                for g in slot_games:
                    played |= {g['home_team'], g['away_team']}
                    hs, as_ = g['home_score'], g['away_score']
                    team_results[g['home_team']][col] = 1 if hs > as_ else (-1 if hs < as_ else 0)
                    team_results[g['away_team']][col] = 1 if as_ > hs else (-1 if as_ < hs else 0)
                for t in TEAMS:
                    if t not in played:
                        team_results[t][col] = None

        else:
            col = f"{mo}.{d}"
            date_cols.append(col)
            played = set()
            for g in day_games:
                played |= {g['home_team'], g['away_team']}
                hs, as_ = g['home_score'], g['away_score']
                team_results[g['home_team']][col] = 1 if hs > as_ else (-1 if hs < as_ else 0)
                team_results[g['away_team']][col] = 1 if as_ > hs else (-1 if as_ < hs else 0)
            for t in TEAMS:
                if t not in played:
                    team_results[t][col] = None

    return date_cols, team_results

# ─────────────────────────── CSV I/O ──────────────────────────────────────────

def load_existing_csv():
    """기존 2026.csv에서 (날짜컬럼 리스트, 팀별 결과 dict) 반환."""
    results = {t: {} for t in TEAMS}
    if not os.path.exists(OUTPUT_FILE):
        return [], results

    with open(OUTPUT_FILE, encoding='utf-8') as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        return [], results

    date_cols = [c.strip() for c in rows[0][1:] if c.strip()]
    for i, team in enumerate(TEAMS):
        row = rows[i + 1] if (i + 1) < len(rows) else []
        for j, col in enumerate(date_cols):
            v = row[j + 1].strip() if (j + 1) < len(row) else ''
            results[team][col] = int(v) if v in ('1', '-1', '0') else None

    return date_cols, results


def write_csv(date_cols: list, team_results: dict):
    """
    CSV를 3개 섹션으로 저장합니다.
      섹션1: 날짜 헤더 + 팀별 개별 경기 결과
      섹션2: 날짜 헤더 + 팀별 누적 승패 마진
      섹션3: 팀별 누적 승률
    """
    # 누적 승패 마진
    cumul = {t: {} for t in TEAMS}
    for team in TEAMS:
        margin = 0
        for col in date_cols:
            r = team_results[team].get(col)
            if r is not None:
                margin += r
            cumul[team][col] = margin

    # 누적 승률 (무승부 제외)
    wr = {t: {} for t in TEAMS}
    for team in TEAMS:
        wins = losses = 0
        for col in date_cols:
            r = team_results[team].get(col)
            if r == 1:   wins   += 1
            elif r == -1: losses += 1
            total = wins + losses
            wr[team][col] = round(wins / total, 3) if total > 0 else None

    rows = []

    # 섹션 1
    rows.append(['날짜'] + date_cols)
    for team in TEAMS:
        row = [team]
        for col in date_cols:
            v = team_results[team].get(col)
            row.append('' if v is None else str(v))
        rows.append(row)
    rows.append([''] * (len(date_cols) + 1))

    # 섹션 2
    rows.append(['날짜'] + date_cols)
    for team in TEAMS:
        rows.append([team] + [str(cumul[team][col]) for col in date_cols])
    rows.append([''] * (len(date_cols) + 1))

    # 섹션 3
    for team in TEAMS:
        row = [team]
        for col in date_cols:
            v = wr[team][col]
            row.append(f"{v:.3f}" if v is not None else '')
        rows.append(row)
    rows.append([''] * (len(date_cols) + 1))

    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(rows)

# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    now = datetime.now()
    today = now.date()
    print(f"KBO {YEAR} 크롤러")
    print(f"  현재: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"  출력: {OUTPUT_FILE}\n")

    existing_cols, existing_results = load_existing_csv()
    existing_set = set(existing_cols)

    # 기존에 있는 날짜의 기본 키 (더블헤더 포함: "3.29.1" → "3.29"로 등록)
    existing_base = {'.'.join(c.split('.')[:2]) for c in existing_cols}

    if existing_cols:
        print(f"기존 데이터: {len(existing_cols)}개 날짜 (마지막: {existing_cols[-1]})\n")
    else:
        print("기존 데이터 없음, 새로 생성합니다.\n")

    all_new_games = []

    for month in MONTHS:
        if date(YEAR, month, 1) > today:
            print(f"[{month}월] 미래 달 - 스킵")
            continue

        print(f"[{month}월] API 호출 중...", end=' ', flush=True)
        try:
            data = fetch_month(YEAR, month)
        except Exception as e:
            print(f"실패: {e}")
            continue

        games = parse_pennant_games(data, YEAR, month, today)

        # 이미 CSV에 있는 날짜(기본 키)는 건너뜀
        new_games = [
            g for g in games
            if f"{g['month']}.{g['day']}" not in existing_base
        ]
        all_new_games.extend(new_games)
        print(f"페넌트레이스 {len(games)}경기 수집, 신규 {len(new_games)}경기")

    if not all_new_games:
        print("\n새로운 경기 없음. 종료.")
        return

    new_cols, new_results = games_to_cols_results(all_new_games)

    # 기존 + 신규 병합
    merged_cols    = list(existing_cols)
    merged_results = {t: dict(existing_results[t]) for t in TEAMS}

    for col in new_cols:
        if col not in existing_set:
            merged_cols.append(col)
    for team in TEAMS:
        for col, val in new_results[team].items():
            if col not in existing_set:
                merged_results[team][col] = val

    merged_cols = sorted(set(merged_cols), key=sort_col)

    print(f"\n총 {len(all_new_games)}경기 추가")
    write_csv(merged_cols, merged_results)
    print("완료.")


if __name__ == '__main__':
    main()
