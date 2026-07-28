# KBO 승패 그래프

KBO(한국 프로야구) 시즌 경기 결과를 매일 자동 수집하고, 팀별 누적 승패 마진을
인터랙티브 그래프로 보여주는 프로젝트입니다.

- 데이터 출처: 다음 스포츠 API (`sports.daum.net/prx/hermes/api/game/schedule.json`)
- 자동화: GitHub Actions가 매일 크롤러를 실행해 `public/2026.csv`를 갱신·커밋
- 시각화: 정적 HTML + Plotly (별도 서버·빌드 불필요)
- 배포: `public/` 디렉토리를 그대로 정적 호스팅 (Cloudflare Pages 등, 빌드 과정 없음)

```
다음 스포츠 API ──▶ crawl_2026.py ──▶ public/2026.csv ──▶ public/index.html
                        ▲                                   (Plotly 그래프)
                GitHub Actions (매일 자동 실행)
```

## 사용법

### 그래프 보기

`fetch()`로 같은 디렉토리의 CSV를 읽기 때문에 로컬에서는 간단한 HTTP 서버가 필요합니다.
반드시 `public/`을 문서 루트로 띄워야 합니다.

```bash
cd public && python3 -m http.server 8000
# 브라우저에서 http://localhost:8000/ 열기
```

CSV 자동 로드에 실패해도 페이지의 "파일 업로드"로 CSV를 직접 올려 볼 수 있습니다.

### 크롤러 수동 실행

```bash
python3 crawl_2026.py
```

- 표준 라이브러리만 사용하므로 추가 패키지 설치가 필요 없습니다.
- 3월~10월을 월 단위로 API 호출하여 **페넌트레이스 완료 경기**만 수집합니다.
  (시범경기·포스트시즌·미경기·취소 경기는 제외)
- 기존 `public/2026.csv`에 없는 날짜만 **증분 병합**하므로 여러 번 실행해도 안전합니다.
- 출력 경로는 스크립트 위치 기준이므로 어느 디렉토리에서 실행하든 `public/2026.csv`에 기록됩니다.

### 자동 실행 (GitHub Actions)

GitHub에 등록된 [crawl_kbo.yml](.github/workflows/crawl_kbo.yml) 워크플로가
매일 1회(UTC 14:00 = KST 23:00) 크롤러를 실행하고, `public/2026.csv`에 변경이 있을 때만
자동으로 커밋·푸시합니다. GitHub 웹의 "Run workflow" 버튼으로 수동 실행도 가능합니다.

정적 호스팅을 붙여 두었다면 이 봇 커밋이 그대로 재배포 트리거가 되므로,
크롤링 → 커밋 → 사이트 갱신이 사람 손 없이 이어집니다.

> cron 표기는 UTC 기준입니다 (KST = UTC + 9시간).
> 실제 스케줄은 GitHub 저장소에 커밋된 워크플로 파일이 기준이므로,
> 로컬 파일만 수정하면 반영되지 않습니다.

## public/2026.csv 형식

한 파일 안에 빈 줄로 구분된 4개 섹션이 순서대로 들어 있습니다.
날짜 컬럼은 `월.일` 형식이고, 더블헤더는 `월.일.1`, `월.일.2`로 나뉩니다.

| 섹션 | 내용 | 형식 |
|---|---|---|
| 1 | 팀별 개별 경기 결과 | 승 `1` / 패 `-1` / 무 `0` / 경기 없음은 빈칸 |
| 2 | 팀별 누적 승패 마진 | (누적 승수 − 누적 패수) |
| 3 | 팀별 누적 승률 | 무승부 제외, 소수 셋째 자리 (예: `0.556`) |
| 4 | 경기별 상세 | `날짜,요일,경기,원정팀,원정점수,홈팀,홈점수` |

- 섹션 1·2는 `날짜,3.28,3.29,...` 헤더로 시작하고, 섹션 3은 헤더 없이 팀 행만 있습니다.
- 팀 행 순서는 `crawl_2026.py`의 `TEAMS` 상수 순서와 같습니다.
- `public/index.html`은 섹션 1~3으로 그래프를 그리고, 섹션 4로 경기별 상세 표를 채웁니다.

### 더블헤더 판별 규칙

한 팀이 하루에 2경기 이상 출전하면 더블헤더로 처리합니다.

1. 시작 시각이 2종류 이상이면: 이른 시각 = 1차전, 나머지 = 2차전
2. 시각이 같으면: 같은 팀 쌍 안에서 `gameId` 순으로 1·2차전 배정

## 파일 구성

```
public/                  ← 정적 호스팅 대상. 이 디렉토리만 공개된다
├── index.html           메인 그래프 페이지
├── 2026.csv             현재 시즌 데이터 (크롤러가 갱신)
└── 2025.csv             지난 시즌 데이터
crawl_2026.py            크롤러
.github/workflows/       자동화
```

| 경로 | 설명 |
|---|---|
| [public/index.html](public/index.html) | **메인 그래프 페이지.** 듀얼 범위 슬라이더·팀 선택·경기별 상세 표 |
| [public/2026.csv](public/2026.csv) | 2026 시즌 데이터 (위 4섹션 형식) |
| [public/2025.csv](public/2025.csv) | 2025 시즌 데이터. 자동 로드 대상은 아니고, 페이지의 "CSV 업로드"로 열어 본다 |
| [crawl_2026.py](crawl_2026.py) | 핵심 크롤러. API 호출 → 파싱 → `public/2026.csv` 증분 갱신 |
| [.github/workflows/crawl_kbo.yml](.github/workflows/crawl_kbo.yml) | 매일 자동 크롤링 워크플로 |

크롤러와 워크플로는 `public/` 밖에 있으므로 배포 시 웹에 노출되지 않습니다.

> 이전 세대 페이지(`clientside3~6.html`, `graph_dashboard.html`)와 UI 시안
> (`test*.html`)은 모두 현재 `public/index.html`의 부분집합이라 정리했습니다.
> 필요하면 git 히스토리에서 되살릴 수 있습니다.

## 새 시즌으로 넘어갈 때

1. [crawl_2026.py](crawl_2026.py)의 `YEAR`(32행), `OUTPUT_FILE`(31행)의 연도
2. 같은 파일의 `TEAMS`(29행) 순서 — 그래프 범례 순서에 영향
3. [public/index.html](public/index.html)의 `defaultCsvName`(1137행)
4. [워크플로](.github/workflows/crawl_kbo.yml)의 `git add` 대상 파일명(28행)
