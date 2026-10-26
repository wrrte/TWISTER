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
`wandb_runs_classification.csv`, `converted_results.xlsx`입니다.
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

CSV에는 게임, 환경, 시드, retrieval 설정, 최종 점수와 평가 정보, 상태, run ID와
링크를 기록합니다. 시드는 config → run 이름의 `seed숫자` → W&B metadata의
`--seed` 순서로 찾습니다. retrieval은 기록된 config와 `_O`/`_X` 분기 접미사를
사용합니다. 과거 TWISTER run은 config를 업로드하지 않았을 수 있으므로, 판단할
근거가 없으면 `Unknown`으로 둡니다. 현재 로컬 defaults를 과거 run에 적용하지
않습니다. `_warmup` run은 공유 중간 단계로 분류하고 history를 조회하지 않습니다.

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

Excel 시트:

- `Results`: 게임·retrieval 설정별 시드 점수와 비교 지표 3개를 표시합니다.
  게임명 아래에는 Random/Human 점수를, 각 게임의 첫 행에는
  `TWISTER (논문): 점수`를 표시합니다. 기준값은 사용자가 첨부한 Atari100k 표
  (2026-10-22)의 Random, Human, TWISTER (ours) 열을 그대로 옮겼습니다.
  이미지에 표시된 정수값을 사용합니다. CSV에 등장한 게임만 표에 표시하며,
  참조표에 없는 환경에는 논문 행을 추가하지 않습니다.
  `Environment` 열은 생략하고 시드 숫자 헤더는 왼쪽 정렬합니다.
  실행 중인 run은 `RUNNING`으로 표시합니다. 셀 메모에서 원본 점수와 run 링크를
  모두 확인할 수 있습니다.
- `Runs`: CSV 전체 원본. 시드 미확인 run도 보존합니다.
- `Missing`: 최종 점수가 없는 run. 실패, 실행 중, warmup 등을 상태로 구분합니다.

오른쪽 비교 지표는 STORM과 같은 행별 비교 방식으로 계산합니다.

- `Mean (공통 시드)`: 해당 게임의 OFF와 모든 ON 설정 양쪽에 완료된 최종 점수가
  있는 공통 시드의 평균입니다. ON 결과가 아직 없으면 완료된 baseline의 모든
  유효 시드로 계산합니다. 같은 설정·시드의 중복 run은 먼저 평균합니다.
- `Δ Score (행별 비교)`: 논문 행에는 `OFF 평균 − TWISTER 논문 점수`를 부호와
  함께 표시합니다. ON 행에는 `|ON 평균 − OFF 평균|`를 표시합니다.
- `Δ HNS (행별 비교)`: 논문 행에는 `(OFF 평균 − 논문 점수) / (Human − Random)`,
  ON 행에는 `(ON 평균 − OFF 평균) / (Human − Random)`을 표시합니다.
  부호를 유지하며 백분율이 아닙니다.

실행 중·점수 누락·시드 미확인은 비교 지표에서 제외하며, 별도의 시드 제외 목록은
사용하지 않습니다. OFF baseline이 확인되지 않거나 ON/OFF 사이 공통 시드가 없으면
비교 지표는 빈칸으로 둡니다. 논문 행의 평균, OFF 행의 두 차이 값도 빈칸입니다.

retrieval config가 없는 과거 run의 정확한 세부 설정 비교에는 당시 설정 정보가
추가로 필요하지만, 최종 점수 자체의 추출에는 필요하지 않습니다.

## 누락된 Atari100k 학습을 TITAN 큐에 추가

```bash
# 미리보기: 파일을 변경하지 않음
python results/append_missing_atari_jobs.py --dry-run

# job_queue_titan.lock을 잡고 기존 큐 끝에 추가
python results/append_missing_atari_jobs.py
```

`converted_results.xlsx`의 `Runs` 시트를 기준으로 Atari100k 전체 26개 게임의
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
