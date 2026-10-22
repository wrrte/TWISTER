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

네트워크 접속 없는 검증:

```bash
python -m unittest discover -s results -p 'test_*.py'
```
