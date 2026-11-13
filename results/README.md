# TWISTER 최종 평가 결과 추출

W&B Public API로 `Evaluation-final/score`를 읽어 CSV와 Excel을 만듭니다.
학습을 실행하거나 W&B 데이터를 수정하지 않습니다.

TWISTER 루트에서 실행:

```bash
python -m pip install -r results/requirements.txt
python results/classify_wandb_runs.py --xlsx
```

기본 대상은 스크린샷의 `choemj-kaist/nnet`입니다. `WANDB_ENTITY`,
`WANDB_PROJECT` 또는 `--entity`, `--project`로 바꿀 수 있습니다.
인증에는 저장된 W&B 로그인 또는 `WANDB_API_KEY`를 사용합니다.
인증이 없다면 터미널에서 먼저 `wandb login`을 실행하세요. API 키를 코드에 넣을
필요가 없습니다. Forge 웹사이트 주소는 API 주소가 아니므로 `--base-url`에 넣지
않습니다. 별도 설치 서버에서는 해당 서버의 API 주소를 지정하세요.

```bash
# 스크린샷의 run만 추출
python results/classify_wandb_runs.py --run-id cfmagzmy --xlsx

# 완료된 run만 / 다른 프로젝트 / 다른 출력 경로
python results/classify_wandb_runs.py --entity choemj-kaist --project nnet \
    --states finished --output results/final_scores.csv --xlsx results/final_scores.xlsx

# 이미 저장한 CSV를 다시 Excel로 변환 (W&B 접속 없음)
python results/convert_csv_to_excel.py
```

출력 기본 경로는 실행 위치와 무관하게 이 디렉터리의
`wandb_runs_classification.csv`, `performance_twister.xlsx`입니다.
`--xlsx`를 생략하면 CSV만 만듭니다. 기존 파일은 성공적으로 저장한 후 교체합니다.

점수는 summary의 정확한 `Evaluation-final/score`를 우선 사용합니다.
값이 없거나 유효한 숫자가 아니면 `scan_history`로 해당 키의 마지막 유효 값을
읽습니다. `eval/*` 또는 TensorBoard의 `Evaluation-final/0/score`로 대체하지
않습니다. `0`과 음수도 유효한 점수입니다. history 조회를 생략하려면
`--no-history-fallback`을 사용하세요.

기본적으로 `callbacks/.../atari100k-*`, `callbacks/.../dmc-*` 형식의 run을
모든 상태에서 읽습니다. 이름 필터는 `--name-regex`로 변경할 수 있고,
`--name-regex ''`로 해제할 수 있습니다. 조회 결과가 없거나 API 호출에 실패하면
기존 CSV를 교체하지 않습니다. 개별 history 조회 오류는 CSV의 `Error`에 기록하고
종료 코드 1을 반환합니다. 아직 점수가 없는 run은 `N/A`로 남깁니다.
실행 중인 run은 `Score Status`를 `running`으로 표시합니다. 이 상태는 점수 유무와
warmup 여부보다 우선하며, 실행 중인 run의 history는 조회하지 않습니다.

CSV에는 게임, 환경, 시드, retrieval 설정, 최종 점수와 평가 정보, 상태, run ID와
링크를 기록합니다. 시드는 config → run 이름의 `seed숫자` → W&B metadata의
`--seed` 순서로 찾습니다. retrieval은 기록된 config와 `_O`/`_X` 분기 접미사를
사용합니다. 과거 TWISTER run은 config를 업로드하지 않았을 수 있으므로, 판단할
근거가 없으면 `Unknown`으로 둡니다. 현재 로컬 defaults를 과거 run에 적용하지
않습니다. 실행 중이 아닌 `_warmup` run은 `warmup`으로 분류하고 history를 조회하지
않습니다. 실행 중인 `_warmup` run은 `running`으로 분류하되 retrieval 설정은 보존합니다.

사용자가 2026-10-22에 확인한 기존 11개 run은 모두 retrieval 미사용 baseline입니다.
이 확인 내용은 `run_overrides.json`에 run ID별로 저장하여 다음 API 추출에도
적용합니다. 다른 run은 이 확인 내용을 자동으로 이어받지 않습니다.
다른 보정 파일은 `--run-overrides PATH`로 지정할 수 있습니다.

2026-10-26에 사용자가 추가 확인한 Road Runner 완료 run도
`run_overrides.json`에 `retrieval_enabled: false`로 기록했습니다.

- `xetdccow`: seed 9999, 최종 점수 17628
- `6pn495jh`: seed 9998, 최종 점수 23434

`convert_csv_to_excel.py`도 이 보정 파일을 읽으므로, 과거 CSV에 두 run이
`Retrieval Unknown`으로 저장되어 있어도 Excel의 `Results`와 `Runs`에서는
`Retrieval OFF`로 반영하고 OFF 평균·비교 지표를 계산합니다. 원본 CSV는
수정하지 않습니다. 다른 보정 파일은 변환 시에도 `--run-overrides PATH`로
지정할 수 있습니다.

`run_overrides.json`은 run ID 외에 전체 run 이름도 키로 사용할 수 있으며,
수동 평가한 `score`, `steps`, `episodes`, `num_envs`, `model_step`, `checkpoint`,
`source`도 지원합니다. 명시한 점수는 CSV/W&B 점수보다 우선하며, Excel 변환 시
CSV에 없는 run은 완료된 평가 행으로 추가합니다. 원본 CSV는 수정하지 않습니다.
점수 출처는 `Score Source`와 셀 메모에 남깁니다.

2026-11-03 Kangaroo seed 9998의 재평가 결과는 **924점**, 100 에피소드,
병렬 환경 32개로 기록했습니다. 같은 파일에 다음 형식으로 결과를 추가한 뒤
`python results/convert_csv_to_excel.py`를 실행하면 됩니다.

```json
"callbacks/kangaroo_seed9998/atari100k-kangaroo": {
  "retrieval_enabled": false,
  "score": 924,
  "episodes": 100,
  "num_envs": 32,
  "model_step": 100000,
  "source": "standalone evaluation, user confirmed 2026-11-03"
}
```

전체 run 이름을 키로 추가하면 해당 seed 열이 자동 생성됩니다. run ID를 키로 쓰면서
CSV에 없는 행을 추가하려면 `env_name`, `seed`, `retrieval_enabled`도 지정해야 합니다. 점수 보정 시
평가 메타데이터는 함께 지정한 값만 사용하며, 생략한 값은 `N/A`로 둡니다.

2026-11-03 사용자가 현재까지 기록된 모든 run을 Retrieval OFF로 확인했습니다.
현재 CSV 및 로컬 W&B 디렉터리에 기록된 run ID를 `run_overrides.json`에 보정해
기존 기록을 OFF로 분류합니다. 이후 생성되는 run은 해당 run의 config를 사용합니다.
다른 GPU(RTX3090)에서 실행한 Gopher seed 3110의 점수는 **33627.2**, 평균 episode
steps는 **58265.52**로 기록했습니다. 이 평가의 에피소드 수와 병렬 환경 수는
확인되지 않았으므로 `N/A`로 둡니다.

이 과거 평가 run(`callbacks/atari100k_seed3110/atari100k-gopher`)만 Excel의
`Results` 시드 열과 상세 시트에서 **`3110-old`**로 표시합니다. 새로 실행한 Gopher
seed 3110의 ON/OFF run은 `3110`으로 유지하며, 과거 점수와 중복 평균하거나
공통 시드로 짝짓지 않습니다. `update_tex.py`도 같은 구분을 적용합니다.
`--all-seed`에서는 시드 번호와 무관하게 비교하므로 `3110-old`도 별도 유효 시드로
포함하고, 기존 점수 순서에 따른 개수 조정 규칙을 적용합니다. 원본 CSV는 보존합니다.

Excel 시트:

- `Results`: 게임·retrieval 설정별 시드 점수와 비교 지표 3개를 표시합니다.
  게임명 아래에는 Random/Human 점수를, 각 게임의 첫 행에는
  `TWISTER (논문): 점수`를 표시합니다. 기준값은 사용자가 첨부한 Atari100k 표
  (2026-10-22)의 Random, Human, TWISTER (ours) 열을 그대로 옮겼습니다.
  이미지에 표시된 정수값을 사용합니다. CSV와 보정 파일에 등장한 게임만 표에 표시하며,
  참조표에 없는 환경에는 논문 행을 추가하지 않습니다.
  `Environment` 열은 생략하고 시드 숫자 헤더는 왼쪽 정렬합니다.
  실행 중인 run은 `RUNNING`으로 표시합니다. 셀 메모에서 원본 점수와 run 링크를
  모두 확인할 수 있습니다.
- `Runs`: 보정된 CSV 행과 추가한 로컬 평가 행. 시드 미확인 run도 보존합니다.
- `Running`: 실행 중인 모든 run. 실행 중인 shared warmup도 포함합니다.
- `Missing`: 실행 중인 run을 제외하고 최종 점수가 없는 run. 실패, warmup 등을
  상태로 구분합니다.

오른쪽 비교 지표는 STORM과 같은 행별 비교 방식으로 계산합니다.

- `Mean (결과 쌍)`: 해당 게임의 OFF와 모든 ON 설정 양쪽에 완료된 최종 점수가
  있는 동일 시드 쌍과 아래의 명시적 baseline 재사용 쌍의 평균입니다. ON 결과가 아직 없으면 완료된 baseline의 모든
  유효 시드로 계산합니다. 같은 설정·시드의 중복 run은 먼저 평균합니다.
- `Δ Score (행별 비교)`: 논문 행에는 `OFF 평균 − TWISTER 논문 점수`를 부호와
  함께 표시합니다. ON 행에는 `|ON 평균 − OFF 평균|`를 표시합니다.
- `Δ HNS (행별 비교)`: 논문 행에는 `(OFF 평균 − 논문 점수) / (Human − Random)`,
  ON 행에는 `(ON 평균 − OFF 평균) / (Human − Random)`을 표시합니다.
  부호를 유지하며 백분율이 아닙니다.

실행 중·점수 누락·시드 미확인은 비교 지표에서 제외하며, 별도의 시드 제외 목록은
사용하지 않습니다. OFF baseline이 확인되지 않거나 유효한 동일 시드/명시적 재사용 쌍이 없으면
비교 지표는 빈칸으로 둡니다. 논문 행의 평균, OFF 행의 두 차이 값도 빈칸입니다.

retrieval config가 없는 과거 run의 정확한 세부 설정 비교에는 당시 설정 정보가
추가로 필요하지만, 최종 점수 자체의 추출에는 필요하지 않습니다.

### 시드 번호와 무관하게 개수를 맞춰 비교

두 스크립트에 `--all-seed`를 붙이면 공통 시드 대신 게임별로 **OFF와 모든 ON
설정의 유효 시드 수 중 최솟값**에 맞춰 비교합니다. 옵션이 없으면 동일 시드/명시적 재사용 쌍
방식을 사용합니다.

```bash
python results/convert_csv_to_excel.py --all-seed
python results/update_tex.py --all-seed --dry-run
python results/update_tex.py --all-seed
```

같은 설정·시드의 중복 완료 run을 먼저 평균한 뒤, 시드가 더 많은 설정에서
**점수 최솟값 → 최댓값 → 다음 최솟값 → 다음 최댓값 → …** 순서로 초과 시드를
제외합니다. 예를 들어 OFF 점수가 `[10, 20, 30, 40, 50]`이고 ON이 2개이면,
OFF에서 `10 → 50 → 20`을 제외하고 `[30, 40]`으로 계산합니다. 시드 수가 같으면
모든 시드를 사용하며, 양쪽 시드 번호가 전혀 겹치지 않아도 비교합니다. 점수가
같으면 시드 번호 오름차순으로 정렬한 뒤 양 끝에서 같은 순서로 제외합니다.

Excel 평균 열은 `Mean (시드 수 맞춤)`으로 표시하고, 셀 메모에 설정별 사용 시드와
제외 시드의 제거 순서를 기록합니다. 개별 시드 점수 열과 원본 run은 보존하며,
평균과 Δ Score·Δ HNS만 선택한 시드로 계산합니다. TeX 갱신기도 같은 시드를
사용하여 게임 점수와 하단 지표 전체(IQM 포함)를 계산하고, 실행 로그에 OFF/ON
각각의 사용 시드를 표시합니다.

실행 중·실패·warmup·점수 누락·시드 미확인 run은 시드 수에서 제외합니다.
비교할 설정 중 유효 시드가 0개이면 해당 게임의 비교를 생략합니다. ON 설정이
여러 개이면 `update_tex.py --on-config`로 표시할 설정을 선택해도 시드 수는
Excel과 동일하게 모든 ON 설정을 포함한 최솟값에 맞춥니다. ON 설정이 없는 게임은
기존처럼 Excel에서 OFF의 모든 유효 시드로 평균하고, TeX 비교에서는 제외합니다.

## 누락된 Atari100k 학습을 TITAN 큐에 추가

```bash
# 미리보기: 파일을 변경하지 않음
python results/append_missing_atari_jobs.py --dry-run

# job_queue_titan.lock을 잡고 기존 큐 끝에 추가
python results/append_missing_atari_jobs.py
```

`performance_twister.xlsx`의 `Runs` 시트를 기준으로 Atari100k 전체 26개 게임의
시드 `9997, 9998, 9999`를 채웁니다. 엑셀에 없는 게임도 포함합니다.
유효한 최종 점수가 있는 `finished` run은 완료로, `running` run과 기존 큐의
학습 명령어는 확보 예정으로 계산해 중복 추가하지 않습니다. 사용자의 확인에
따라 `Retrieval Unknown` 기록도 인정하며, ON·공유 warmup은 제외합니다.
최종 점수가 없는 완료 run과 실패 run은 해당 시드를 다시 추가합니다.

이미 확보했거나 확보 예정인 시드 수를 반영해 모든 게임에 1개 → 2개 → 3개
시드를 확보하는 순서로 새 작업을 배치합니다. 각 게임에서는 아직 없는 시드 중
`9997, 9998, 9999` 순서로 선택합니다. 기존 큐의 순서는 유지하며, 여러 worker가
실행될 때 단계별 완료를 기다리는 동기화 기능은 없습니다.

환경명은 실제 Atari ROM 이름(`bank_heist`, `road_runner` 등)을 사용하고,
명령어의 나머지 부분은 항상 다음과 같습니다.

```bash
env_name=atari100k-road_runner override_config='{"retrieval_enabled":false}' python3 -u main.py --seed 9997
```

입력·출력 경로는 `--xlsx PATH`, `--queue PATH`로 바꿀 수 있습니다. 기본 경로는
실행 위치와 무관하게 이 저장소의 결과 엑셀과 TITAN 큐입니다. 실행 중인 작업이
엑셀에 반영되지 않은 경우에는 중복을 판단할 수 없으므로 최신 결과 엑셀을
사용하세요. 기본 `python`에 openpyxl이 없다면 `twister` 환경의 Python을 사용하세요.

네트워크 접속 없는 검증:

```bash
python -m unittest discover -s results -p 'test_*.py'
```

## 999N baseline 재사용 목록

[`reusable_baselines.json`](reusable_baselines.json)은 현재 엑셀의 `Runs` 시트에서 만든
수정 가능한 목록입니다. 완료된 9990–9999 OFF 시드 중, 기존 동일 시드 ON/OFF 쌍의
**OFF 점수 평균 이상**인 미사용 baseline을 선정했습니다. 중복 run은 시드별로 먼저
평균하고, Gopher 및 기존 쌍이 없어 기준 평균을 구할 수 없는 게임은 자동 선정에서
제외합니다. 각 게임의 `paired_baseline_mean`과 `paired_seeds`는 선정 당시의 근거입니다.

초기 목록은 Asterix 9997(1374.5), Chopper Command 9997(694), Frostbite 9997(314),
Kung Fu Master 9997(33290), Ms. Pacman 9997(2479.6), Road Runner 9998(23434)입니다.

`games.<ROM 이름>.baselines`에서 항목을 추가·제거하거나 `enabled: false`로 사용을
끄면 됩니다. 각 항목에는 원래 `seed`, 유한한 `score`, 가능한 경우 `run_ids`,
`run_names`, `run_urls`를 유지하세요. 배열 순서가 사용 순서입니다. 평균 기준은 초기
선정에만 사용하고, worker는 이후 사용자가 편집한 목록을 그대로 따릅니다.

새 worker는 조건을 만족한 full FLASH에 미사용 항목 하나를 연결하고 OFF 재학습을
생략합니다. 예전에 건너뛴 full FLASH도 재검사 시 조건을 통과하면 연결할 수 있으며,
원래 warmup이 없어도 됩니다. 한 baseline은 게임 내 한 쌍에만 쓰고, 연결은
`reproduction/baseline_pairings/<game>/seed<ON seed>.json`에 원본 999N 시드·점수·run과
ON 시드·점수·통과 조건을 함께 저장합니다. 서로 다른 시드임을 유지하며 가상의 OFF
run을 만들지 않습니다. 이미 확정한 연결은 목록을 수정해도 바꾸지 않습니다.

같은 서버의 여러 worker/campaign은 공통 잠금으로 중복 사용을 막습니다. 공유 게임은
서버 간 중복 할당을 피하려고 기존 이력 담당 서버만 목록을 소비합니다(Krull은 B).
Gopher는 수동으로 목록에 넣어도 재사용하지 않습니다. 일일 `reproduction_cluster.py sync`는
자기 서버가 만든 연결 파일도 보고서와 함께 교환합니다.

`update_tex.py`와 Excel 변환기는 이 연결 파일을 자동으로 읽어 해당 OFF/ON 점수를
같은 쌍으로 집계합니다. 출력과 셀 메모에는 `OFF 9997 -> ON 3130`처럼 실제 두 시드를
표시합니다. CSV에 옛 OFF run이 없어도 연결 파일의 확정 점수를 사용할 수 있습니다.
일반 동일 시드 쌍과 재사용 쌍을 중복 집계하지 않으며 `--all-seed`에서도 실제 OFF
시드 수를 늘리는 가상 별칭은 만들지 않습니다. 연결된 원본 OFF 시드는 일반 동일 시드
쌍으로 다시 소비하지 않고, 명시적 연결이 있는 ON 시드는 그 연결을 우선합니다.

```bash
# 원본 엑셀로 목록을 다시 만들 때만 실행. 기존 수동 편집을 보호합니다.
python results/baseline_pairs.py --workbook results/performance_twister.xlsx
# 기존 목록을 교체하려면 명시적으로 --overwrite 추가

# 연결 파일의 다른 경로를 시험하려면 두 집계기에 같은 경로 지정
python results/update_tex.py --baseline-pairings reproduction/baseline_pairings --dry-run
python results/convert_csv_to_excel.py --baseline-pairings reproduction/baseline_pairings
```

## 논문의 main performance 표 갱신

TWISTER 루트에서 최신 CSV를 만든 뒤 실행합니다. TeX 갱신 자체는 W&B에 접속하지
않으며, CSV·Excel 파일이나 논문의 본문·다른 표·다른 방법의 점수를 수정하지 않습니다.

```bash
python results/classify_wandb_runs.py --xlsx
python results/update_tex.py

# 파일을 수정하지 않고 공통 시드와 하단 지표 확인
python results/update_tex.py --dry-run

# 입력 경로와 보정 파일 지정
python results/update_tex.py --csv results/wandb_runs_classification.csv \
    --tex ../iclr2027_conference.tex --run-overrides results/run_overrides.json
```

`results` 폴더 안에서는 다음과 같이 실행합니다. 기본 CSV·TeX 경로는 스크립트 위치를
기준으로 정하므로 실행 폴더와 무관합니다. 사용 중인 TeX의 절대 경로도 출력합니다.

```bash
python update_tex.py --all-seed
python update_tex.py --all-seed --dry-run
```

기본 대상은 TWISTER 루트 기준 `../iclr2027_conference.tex`의 `tab:main_performance`입니다.
`TWISTER`, `TWISTER+ours`, `$\Delta$` 세 열이 있으면 해당 데이터 셀을 갱신합니다.
없으면 표 끝에 세 열을 추가하고 `tabular`의 열 지정도 함께 확장합니다.
Random/Human·STORM·DRAMA 셀과 제목·다른 표는 보존합니다.
기존 STORM·DRAMA 갱신기도 이 세 열이 추가된 배치를 허용하도록 열 검사만
확장했습니다. 각 갱신기는 자기 방법의 세 열만 갱신합니다.

기본 점수는 Excel의 **Mean (결과 쌍)**와 같은 방식으로 집계합니다.
`--all-seed` 사용 시에는 위의 **시드 수 맞춤** 방식을 사용하며, 아래 HNS 계산식은
각 설정에서 선택한 시드에 적용합니다.

- 보정 파일을 먼저 적용하고, 완료된 `finished` run의 유한한 최종 점수만 사용합니다.
  실행 중·실패·warmup·점수 누락·시드 미확인 run은 사용하지 않습니다.
- 같은 게임·설정·시드의 중복 run은 먼저 평균합니다. 그다음 OFF와 모든 ON 설정에
  동일 시드 쌍과 명시적 baseline 재사용 쌍을 골라 평균합니다. baseline만 있는 게임은 사용하지 않습니다.
- ON 설정이 여러 종류이면 `--on-config 'CSV의 정확한 Config 문자열'`로 표에 넣을
  설정을 지정합니다. 공통 시드는 Excel과 마찬가지로 모든 ON 설정과 교집합을 취합니다.
- 유효한 결과 쌍이 없는 게임은 `-`로 표시하고 집계에서 제외합니다. 기존 숫자를
  남겨 두거나 누락을 0점으로 계산하지 않습니다. 게임·시드의 현재 포함 범위를 출력합니다.

HNS에는 **논문 표에 있는** Random/Human 값을 사용합니다. Excel의 참고용 정수
Random/Human 값으로 기존 표를 덮어쓰지 않습니다. 각 시드의 점수를 $A_{g,s}$라 할 때
$h_{g,s}=(A_{g,s}-R_g)/(H_g-R_g)$, 게임별 평균을 $\bar h_g$로 놓고 다음을 계산합니다.

| 하단 지표 | 계산식 |
| --- | --- |
| #Superhuman | $\sum_g \mathbf{1}[\bar h_g>1]$ |
| Mean | $\operatorname{mean}_g \bar h_g$ |
| Median | $\operatorname{median}_g \bar h_g$ |
| IQM | 공통 시드의 $h_{g,s}$를 모두 모아 정렬한 뒤 양쪽에서 $\lfloor N/4\rfloor$개씩 제외한 평균. 기존 STORM 갱신기와 같은 trimming 방식입니다. |
| Optimality Gap | $\operatorname{mean}_g\max(0,1-\bar h_g)$ |

Mean·Median·Optimality Gap은 게임마다 같은 가중치를 주며, IQM은 시드별 HNS를
pooling하므로 시드 쌍이 많은 게임이 더 많은 관측치를 제공합니다. 계산에는 표시용으로
반올림하기 전 값을 사용합니다. 게임 점수는 소수점 한 자리까지, 하단 HNS 지표는
세 자리까지 표시합니다. $\Delta$는 표시된 ON−OFF를 정확히 빼서 계산하며, Optimality
Gap은 감소가 개선이고 나머지는 증가가 개선입니다.
