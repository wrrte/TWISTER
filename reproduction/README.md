# GPU 4개 / 4개 / 1개를 세 서버에서 사용하기

`N_train.sh`는 GPU별 관리 프로세스가 `reproduce.py --once` worker를 반복 실행합니다. 각 서버의 worker는
로컬 `state.json`을 공유합니다. 서버 간에는 `servers.json`의 게임 배정과
`reports/a.json`, `b.json`, `c.json`의 작은 결과 요약을 Git으로 교환합니다.
담당 서버를 바꿀 때는 `handoffs/<game>.json`으로 점수·시드 이력도 교환합니다.
`job_queue_3090.txt`와 다른 `job_queue_*.txt`는 이 실행에서 사용하지 않습니다.

기본 배정은 A(4 GPU) 11게임, B(4 GPU) 13게임, C(1 GPU) 2게임입니다.
`servers.json`의 `shared_games`에 등록한 Gopher와 Krull은 **모든 서버의 모든 GPU**에서
실행할 수 있습니다. 공유 배정을 포함한 담당 목록은 A 13게임, B 14게임, C 3게임입니다.
Gopher의 기존 이력 담당은 C, Krull은 B이며, Road Runner와 Up N Down도 B 담당입니다.
서버 A는 RTX 3090 서버, B는 RTX A6000 서버, C는 나머지 1-GPU 서버입니다.
게임 배정은 이 역할 등록으로 결정하며, 학습 시작 시 선택한 GPU 모델과
PyTorch의 CUDA 인식도 검사합니다. **완료된 성공 쌍이 3개 미만인 게임을 최우선**으로
배정하고, 3개를 확보한 게임의 네 번째 쌍은 가장 낮은 우선순위로 배정합니다.
같은 우선순위에서는 실행 중인 작업이 적은 게임, 시도 횟수가 적은 게임,
담당 목록의 순서로 배정합니다. 실행 중인 시도는 아직 완료된 성공 쌍으로 세지 않습니다.

**모든 담당 게임**이 같은 서버의 여러 GPU에서 서로 다른 시드를 동시에 실행할 수
있습니다. 먼저 3쌍 미만인 게임들에 GPU를 나누고, 그런 게임이 GPU 수보다 적으면
같은 게임을 중복 배정합니다. 예를 들어 A의 GPU 4·5·6·7에서 다른 담당 게임이 모두
3쌍이고 Alien만 2쌍이면, 네 GPU 모두 Alien의 다른 시드를 실행합니다. 다른 게임의
네 번째 쌍이나 그 쌍의 중단된 작업·과거 baseline 복구보다 이 배정이 우선합니다.
담당 게임이 모두 3쌍이면 네 번째 쌍을 채우며, 이 단계의 새 작업은 게임당
로컬에서 최대 한 개씩 배정합니다. Gopher·Krull의 3쌍 판정에도 전체 서버 합계를 쓰고,
다른 게임의 담당 서버와 시드 구분은 그대로 유지합니다.
게임별 시드는 `3090, 3100, 3110, ...` 순서이며 기존에 사용한 시드는 건너뜁니다.
**성공한 baseline–full FLASH 쌍이 누적 4개가 되면 해당 게임의 학습을 종료합니다.**
조건 2로 완성한 쌍과 산출물이 정리된 과거 쌍, 다른 서버에서 넘겨받은 완료 쌍도
4개에 포함합니다. baseline 점수 자체는 성공 여부에 영향을 주지 않습니다.

공유 게임은 **각 게임별 전체 서버 합계 성공 4쌍**을 목표로 합니다. 현재 로컬
완료 쌍과 다른 서버의 마지막 Git sync 보고서를 합산해, 4쌍 이상이면 해당 게임의
새 시드·과거 baseline 복구를 배정하지 않습니다. 서버 간 실시간 공유가 없으므로
원격 결과는 마지막 sync를 기준으로 판정합니다.

**완료 쌍과 조건을 통과한 full FLASH 최종 평가를 합쳐 4개가 확보되면 초과 작업을
중단합니다.** 완료 쌍은 유지하고, 아직 baseline이 없는 통과 시드 중 full FLASH 점수가
높은 순서로 남은 자리를 선택합니다. 조건은 아래의 개별 점수 또는 평균 점수 조건이며,
학습 중의 중간 점수로 취소하지 않습니다. 선택된 시드의 baseline은 끝까지 완료합니다.
숫자 이력만 있고 현재 복구 중이 아닌 이관 기록은 baseline 자리를 예약하지 않습니다.

각 worker가 약 1초마다 확인해 초과 warmup·full FLASH·baseline·평가의 자식 프로세스
그룹을 종료한 뒤 다음 게임을 배정받습니다. 취소는 학습 실패와 구분하여 `cancelled`로
기록하고 같은 시드를 다시 배정하지 않습니다. 필요한 네 시드의 baseline이 모두
실행 중이면 다른 배정 가능한 게임을 진행하며, 작업이 없을 때만 기다립니다.
보고서에는 baseline을 기다리는 통과 full FLASH 시드·점수도 포함합니다. 원격 sync 전이나
중단을 확인하기 직전에 이미 끝난 평가가 있으면 최종 쌍이 4개를 넘을 수 있습니다.

보고서의 `successful_pair_seeds`로 같은 시드를 중복 집계하지 않습니다. 이관한 원본
기록은 수신 서버에서만 성공 수에 포함하고, 원래 서버의 보관 이력을 다시 더하지 않습니다.
시드 목록이 없는 기존 보고서는 현재 이력 담당 서버(C의 Gopher, B의 Krull) 것만
합산합니다. baseline 평균 판정은 기존처럼 로컬·이관 이력을 사용하며, 상위 4개
산출물 보관도 각 서버에서 수행합니다.

시드 중복 방지를 위해 Gopher는 A에서 `3091, 3101, ...`, B에서 `3092, 3102, ...`,
C에서 기존 `3090, 3100, ...` 순서를 사용합니다. Krull은 A에서 `3091, 3101, ...`,
B에서 기존 `3090, 3100, ...`, C에서 `3092, 3102, ...`를 사용합니다. 이미 사용·예약된
시드는 계속 건너뜁니다. 이력 담당 서버는 시드 구분의 기준이므로 변경하지 않습니다.

이미 아래 B/C 이관을 완료했다면 추가 handoff나 초기화는 필요하지 않습니다.
**기존 worker는 아래의 자동 전환 절차를 사용하거나, 현재 trial을 마친 뒤 모두 종료하고
새 worker들을 시작하세요.** 이전 worker와
새 worker를 같은 상태 파일에서 섞어 실행하면 이전 코드가 새 공유 작업을 이관 이력으로
옮길 수 있습니다. 새 코드는 기존 released 이력과 구분해 B의 Gopher도 계속 실행합니다.

baseline은 다음 중 하나만 만족하면 실행하거나, 재사용 목록의 미사용 결과로 연결합니다.

1. 이번 full FLASH의 HNS가 논문 HNS − 0.1 이상.
2. 같은 게임에서 기존에 완료한 모든 baseline–full FLASH 쌍의 **full FLASH 점수**에
   이번 full FLASH 점수를 더한 평균의 HNS가 논문 HNS + 0.1 이상.

평균은 상위 4개 보관 여부와 관계없이 모든 완료 쌍을 포함합니다. baseline이 없는
시드, 다른 게임, baseline 점수는 제외합니다. 판정 시 사용한 시드·평균과 통과
사유를 trial의 `baseline_gate`에 저장합니다. 통과한 판정은 재개할 때 유지하며,
과거에 탈락한 판정은 새로운 완료 쌍이 생기면 다시 계산합니다.
산출물은 현재 서버에서 full FLASH 점수 상위 4개를 보관하되, 이번 시드의
조건부 baseline이 끝나기 전에 warmup을 삭제하지 않습니다.

`results/reusable_baselines.json`의 미사용 999N baseline이 있으면 새 OFF 학습보다 먼저
해당 점수를 연결합니다. 원래 baseline 시드와 ON 시드는 별도로 보존하고, 연결 즉시
완료 쌍으로 세어 평균·3쌍 우선순위·4쌍 종료 판정에 포함합니다. Gopher는 제외합니다.
공유 게임의 목록은 기존 이력 담당 서버(Krull은 B)에서만 소비하며, 같은 서버의 여러
GPU는 잠금으로 각 항목을 한 번씩만 사용합니다. 연결 파일도 일일 sync로 공유하므로
다른 서버의 `results/update_tex.py`에서도 반영됩니다. 목록 편집과 선정 기준은
[결과 집계 안내](../results/README.md#999n-baseline-재사용-목록)를 참고하세요.

## 과거에 건너뛴 baseline 다시 실행

게임이 쉬는 시점마다, 새 시드를 시작하기 전에 baseline이 없는 기존 full FLASH
결과를 점수가 높은 순서로 검사합니다. 새로 완료한 쌍 덕분에 조건 2를 만족하게
되었다면, 필요한 파일이 있는 가장 높은 점수의 시드부터 baseline을 실행합니다.
새 worker를 시작했을 때도 기존 탈락 기록을 재검사하므로 이전 코드에서 건너뛴
시드도 대상입니다. 추가 옵션이나 수동 재검사 명령은 필요하지 않습니다.

- 기존 full FLASH 점수와 시드는 그대로 유지하며 시도 횟수를 늘리지 않습니다.
- 원래 shared warmup의 checkpoint, optimizer·RNG, replay 파일과 run/config 정보가
  있으면 그 warmup에서 OFF 분기만 실행합니다. ON 최종 가중치를 OFF 초기값으로
  사용하지 않습니다.
- 전체 예산을 학습한 호환되는 OFF warmup이 남아 있으면 추가 학습 없이 평가합니다.
- 파일이 없거나 호환되지 않으면 사유를 기록하고 다음 후보를 확인합니다. 과거
  warmup/full FLASH를 처음부터 다시 학습하지 않습니다.
- 하나를 완료할 때마다 완료 쌍 전체의 평균을 다시 계산합니다. 다음 후보도
  통과하면 이어서 처리하고, 성공 쌍이 4개가 되는 즉시 종료합니다.
- 지금 복구할 수 있는 후보는 처리 전에 상위 4개 정리로 삭제하지 않습니다.
  따라서 복구를 기다리는 동안 보관 디렉터리가 일시적으로 4개를 넘을 수 있습니다.

서버 이관 기록에는 점수·시드만 포함됩니다. 예전 서버에만 있는 checkpoint는
새 서버에서 자동으로 읽을 수 없으며, 원래 trial 디렉터리와 참조 replay 경로가
현재 서버에서도 접근 가능할 때만 복구합니다. 숫자 기록만 있는 경우는 건너뜁니다.

## 기존 B/C 배정 변경 적용

기존 상태를 초기화하거나 `setup --reset`을 실행하지 않습니다. 원래 서버의
체크포인트·shared replay 경로는 보존하고 숫자 기록만 새 담당 서버에 넘깁니다.
수신 서버는 완료 쌍을 평균·종료 판정에 포함하고, 미완료 시드도 사용한 시드로
예약하여 새로운 시드에서 학습을 시작합니다. 미완료 분기의 체크포인트를 다른
서버에서 이어 학습하는 방식은 아닙니다.

엑셀은 Git에서 제외되어 있으므로, 현재 엑셀에 있는 시드 목록을
`seed_reservations.json`에도 기록했습니다. 로컬 엑셀이 있으면 그 시드까지 합쳐
중복을 피합니다. Gopher의 `3090, 3100, 3110, 3120, 9997`을 예약하며
`3110-old`도 3110으로 보호합니다. 현재 기록 기준 다음 Gopher 시드는 3130이며,
B에서 더 진행했다면 handoff에 포함된 실제 이력 이후의 시드를 선택합니다.
엑셀 자체와 과거 평가 결과는 수정하지 않습니다.

아래 명령은 TWISTER 디렉터리에서 의존성이 설치된 Python 환경으로 실행합니다.

1. **C에서** Seaquest의 현재 trial(조건부 baseline까지)을 마친 뒤 기존 worker를
   종료합니다. 아래 handoff 자체는 Seaquest가 실행 중이어도 가능하며,
   넘길 두 게임이 실제 실행 중이면 변경 없이 거절합니다.

   ```bash
   python reproduction_cluster.py handoff
   git add reproduce.py reproduction_cluster.py reproduction_seeds.py reproduction_recovery.py README.md reproduction/README.md reproduction/servers.json reproduction/seed_reservations.json reproduction/test_reproduction.py reproduction/handoffs
   git commit -m "Update reproduction gates and reassign B/C games"
   git push
   ```

2. **B에서 Gopher worker만 종료**한 뒤 최신 코드와 C의 숫자 이력을 받습니다.
   다른 게임의 worker는 현재 실험을 계속 진행해도 됩니다. `handoff`는 넘기는
   Gopher의 lease만 확인하며, 이관 후에도 다른 게임의 진행 기록/잠금은 유지합니다.
   가능하면 진행 중인 Gopher trial을 마친 뒤 종료하세요. 중단된 채 넘겨도
   기존 파일은 보존하며, 그 시드는 C에서 다시 배정하지 않습니다.

   ```bash
   git pull --rebase
   python reproduction_cluster.py handoff
   python reproduction_cluster.py sync
   ```

   `sync`는 B의 Gopher 이력을 커밋·push하고, B에 Road Runner와 Up N Down을
   반영합니다. 다른 worker들도 현재 trial을 마친 뒤 모두 종료하고, 이전 버전이
   남아 있지 않을 때 필요한 `N_train.sh`들을 다시 실행하세요. 새 담당 게임과
   공유 게임, baseline 조건·과거 baseline 재검사가 적용됩니다. 이미 실행 중인
   Python worker의 코드는 자동 교체되지 않습니다.

3. **C에서** B가 공개한 Gopher 이력을 받아 배정을 적용하고 재시작합니다.

   ```bash
   git pull --rebase
   python reproduction_cluster.py sync
   ./0_train.sh
   ```

   C의 기본 담당은 Seaquest와 Gopher이며, 공유 게임 Krull도 실행합니다. 아직 완료하지
   않은 trial이 있으면 먼저 재개하고, 이후에는 모든 게임에 같은 우선순위를 적용합니다.
   상대 서버의 handoff 파일이 없으면 학습 시작 전에 필요한 서버/명령을
   안내하고 종료하므로, 오래된 보고서만 보고 시드를 다시 시작하지 않습니다.

4. **A에서도** 기존 worker가 현재 trial을 마친 뒤 모두 종료하고 `git pull --rebase` 후 재시작하면
   새 조건과 과거 baseline 재검사, 누적 4쌍 종료 기준이 적용됩니다.
   A의 기존 11게임도 유지하며, Gopher와 Krull을 공유 게임으로 추가 실행합니다.

학습 없이 검증하려면:

```bash
python -m unittest discover -s reproduction -p 'test_*.py' -v
```

## 최초 한 번

기존 배정 변경은 위 절차를 사용합니다. 새 B/C 설치는 원래 서버의 handoff
파일까지 배포한 뒤 등록합니다.

이 코드와 `servers.json`을 같은 Git 브랜치로 세 서버에 배포한 뒤 TWISTER
의존성과 ROM이 설치된 Python 환경에서 실행합니다. 아래 a/b/c는 GPU 번호가
아니라 서버 이름이며, 각각 다른 서버에 지정해야 합니다.

```bash
# 서버 A에서 한 번 (기존 진행 중인 Alien/Amidar 기록과 경로를 유지)
python reproduction_cluster.py setup a

# 서버 B에서 한 번
python reproduction_cluster.py setup b

# 서버 C에서 한 번
python reproduction_cluster.py setup c
```

설정은 Git에서 제외한 `reproduction.local.json`에 저장됩니다. 이후에는
`N_train.sh`에 서버 이름이나 게임 목록을 반복해서 줄 필요가 없습니다.
커스텀 상태 경로를 원하면 setup 시 `--state PATH`를 지정합니다.
`setup`은 서버 역할 등록이며 학습을 실행하지 않습니다. 등록하지 않은 상태에서
`N_train.sh`를 실행하면 모델을 생성하기 전에 설정 명령을 안내하고 종료합니다.

A6000에서 등록 전에 잘못 시작한 학습을 지우고 B로 새로 시작하려면 모든
worker를 중단한 상태에서 다음을 실행합니다. 이 옵션은 현재 로컬 탐색의
상태와 소유권이 확인된 산출물만 삭제하며, 다른 탐색의 산출물은 유지합니다.

```bash
python reproduction_cluster.py setup b --reset
```

GPU 인식은 학습 없이 다음 명령으로 확인할 수 있습니다.

```bash
python reproduction_cluster.py doctor --gpu 0
```

호스트, 실제 GPU 모델/번호/UUID, 등록한 역할, Python 실행 파일과 PyTorch CUDA
상태를 표시합니다. B 서버의 GPU 0이면 `NVIDIA RTX A6000`과 `cuda:0` 매핑이
확인되어야 합니다. GPU UUID로 실제 장치를 고정하므로 동일 모델 GPU가 여러
개 있어도 번호 혼동을 방지합니다. CUDA를 사용할 수 없거나 역할에 맞지 않는
GPU이면 `N_train.sh`는 학습을 시작하지 않습니다.

서버 A와 B에서는 사용 가능한 GPU 번호에 맞춰 네 worker를 각각 실행합니다.
예를 들어 GPU 0, 1, 2, 3을 사용할 경우 터미널/tmux 창마다 다음 중 하나를
실행합니다. 서버 C에서는 `./0_train.sh` 하나만 실행하면 됩니다.

```bash
./0_train.sh
./1_train.sh
./2_train.sh
./3_train.sh
```

다른 GPU 번호를 사용한다면 해당 `N_train.sh`를 선택하세요. 각 서버의 GPU 번호는
그 서버의 로컬 번호입니다. Python 환경을 명시하려면 `TWISTER_PYTHON`을 설정합니다.
새 worker는 배포된 최신 코드를 사용합니다. 관리 프로세스는 한 시드의 warmup·full FLASH·
조건부 baseline을 처리할 때마다 새 Python worker를 실행합니다. 다음 작업부터 변경한
스케줄러·baseline 재사용 코드가 적용됩니다. 기존 `--once`, `--status`, `--help`도 유지합니다.
전체 목표 달성 시 관리 프로세스도 종료하며, 대기 시 10초 기다리고 오류는 재시도 없이 종료합니다.
저장된 warmup/평가는 재사용하며 기존 상태와 파라미터의 초기화는 필요하지 않습니다.

## CPU 부하에 따른 학습 자동 정지·재개

`N_train.sh`로 시작한 기존 학습을 종료하지 않고, 별도 터미널/tmux에서 서버당
감시 프로그램 하나를 실행합니다. 이후 시작되는 같은 체크아웃의 worker도 감시합니다.

```bash
# 대상 확인만 수행. 학습에 신호를 보내지 않음
python reproduction_cpu_guard.py --dry-run --once

# 자기 계정의 이 TWISTER 체크아웃 학습을 자동 조절
python reproduction_cpu_guard.py

# 필요하면 대상 GPU와 동시 실행 상한도 지정
python reproduction_cpu_guard.py --gpus 1 2 3 4 5 7 --max-running 4
```

동작은 다음과 같습니다.

- 로그의 tqdm 속도를 읽고 `s/it`도 `it/s`로 환산합니다. 처음 연결하거나 재개한 뒤
  60초는 관찰만 하며, **1.30it/s 이하가 30초 동안 유지**되어야 정지 후보가 됩니다.
  `--grace-seconds`, `--slow-seconds`, `--min-rate`로 변경할 수 있습니다.
- 전용 세션으로 실행된 학습 프로세스 그룹에 `SIGSTOP`을 보냅니다. 같은 그룹의 환경
  프로세스도 정지하며, 모델·optimizer·replay 상태와 RAM/VRAM은 유지됩니다.
- 정지 후 **무작위 5~30초**를 기다려 전체 CPU 사용률을 확인합니다. **80% 이하**이면
  한 작업을 재개하고, 초과하면 **30초 뒤** 다시 확인합니다. `--resume-cpu`로 문턱을
  바꿀 수 있습니다. 재개 후에는 30초 동안 추가 조정을 보류합니다. 다른 작업 조정과
  겹치면 개별 작업의 확인 시점도 뒤로 밀릴 수 있습니다.
- 한 번에 한 작업만 조정하고, 감시 중인 학습의 마지막 실행 작업은 속도만으로 정지하지
  않습니다. epoch가 끝난 뒤 평가 속도는 정지 판단에서 제외합니다. 다른 사용자 작업과
  이 체크아웃 외부의 작업, 수동으로 정지한 작업은 제어하지 않습니다.
- CPU 수치만으로 실행 개수가 정확히 4개로 유지되지는 않습니다. 80%에서 하나를 추가해
  다시 포화될 수도 있습니다. `--max-running 4`를 지정하면 감시 대상 중 실행 중인 학습을
  최대 네 개로 제한하며, 느려지면 그보다 적게 실행할 수 있습니다. 최종 평가·다른 사용자
  프로세스는 이 개수에 포함되지 않지만 전체 CPU 수치에는 반영됩니다.

`Ctrl+C`나 정상적인 종료 신호를 받으면 감시 프로그램이 정지시킨 그룹만 재개합니다.
강제 종료(`SIGKILL`) 등으로 재개 처리가 실행되지 않았다면 다음 명령을 사용합니다.
기록된 PID의 시작 시각과 학습 프로세스 신원을 확인하므로 재사용된 PID에는 신호를
보내지 않습니다. 감시 프로그램을 다시 실행해도 이전 정지를 먼저 복구합니다.

```bash
python reproduction_cpu_guard.py --recover
```

## 이미 정지한 worker 자동 전환

이전 버전의 부모 `reproduce.py` PID만 `SIGSTOP`으로 정지해 둔 경우, 다시 재개할 필요 없이
다음 도구를 실행합니다. GPU 번호는 해당 서버의 실제 번호입니다.

```bash
# 부모·현재 자식·게임·단계를 읽기 전용으로 확인
python reproduction_supervisor.py adopt --gpus 4 5 6 7 --inspect

# 이미 정지한 부모와 현재 학습을 인계. 터미널을 유지하거나 tmux에서 실행
python reproduction_supervisor.py adopt --gpus 4 5 6 7
```

각 GPU를 독립적으로 1초마다 확인합니다. 총 메모리가 **1000MiB 미만으로 연속 5초**
유지되고, 원래 자식이 정상 종료해 해당 단계의 결과·체크포인트가 저장되었으며,
그 GPU에 실행 중인 compute 프로세스가 없어야 전환합니다. 메모리가 다시 1000MiB
이상이 되거나 GPU 조회가 끊기면 시간을 다시 셉니다. CPU에서 결과 저장/W&B 정리를
계속하는 자식이 있으면 메모리가 줄어도 기다립니다.

도구는 완료한 자식을 확인한 뒤에만 정지한 부모에 종료 신호를 예약하고 재개하여
기존 종료 처리로 프로세스 그룹과 잠금을 정리합니다. `CONT`만 보내 구버전 학습을
이어가지 않습니다. PID의 시작 시각도 확인하며, 학습 실패·잘못된 결과·남은 잠금이
있으면 해당 GPU의 전환을 중단하고 다른 GPU들은 계속 감시합니다. warmup의 정상적인
공유 경계 생성뿐 아니라, 기존의 종료 코드 75 fallback도 상태에 보존합니다.

완료한 현재 **한 단계**를 같은 상태 파일에 반영하고, 해당 GPU의 새 관리 프로세스를
분리 실행합니다. baseline이 끝났으면 완료 쌍으로 세고, full FLASH가 끝났으면 새 로직이
baseline 재사용 여부부터 판단합니다. 상태 초기화나 `--reset`은 사용하지 않습니다.
인계 중인 trial은 다른 GPU가 먼저 가져가지 못하게 보호합니다. 구버전 부모는 모두
정지해 다음 배정을 할 수 없는 상태여야 하며, 도구는 실행 중인 부모를 자동 정지하지 않습니다.

GPU별 실행 로그는 `results/reproduction/supervisors/gpu4.log` 등이며,
전환 상태는 `gpu4.transition.json` 등에 저장됩니다. 도구를 다시 실행하면 살아 있는
교체 관리 프로세스는 중복 실행하지 않고, 완료 결과 확인 후 중단된 인계는 재개할 수 있습니다.
관리 프로세스와 전환 도구는 GPU UUID별 잠금을 공유합니다.

## 남는 GPU에서 한 번만 돕기

예를 들어 같은 서버의 GPU 5를 잠시 사용할 수 있다면:

```bash
./5_train.sh --once
```

기존 worker들과 같은 상태 파일에서 다음 작업 한 건을 중복 없이 가져옵니다.
shared warmup → full FLASH 학습/최종 평가 → 두 조건 중 하나를 만족하면 같은
warmup에서 baseline 실행까지 마친 뒤 종료합니다. 두 조건 모두 미달이면 baseline을
생략하고 종료합니다. 결과는 정규 실행과 같은 완료/성공 통계, W&B 설정,
게임별 상위 4개 보관 규칙에 포함되며 다음 sync에서 공유됩니다.

예를 들어 담당 게임 중 일부 시도가 완료됐고 다른 게임은 다른 GPU에서 진행 중이라면,
완료된 게임 중 성공 시드 4개가 아직 없는 게임의 다음 시드를 실행합니다.
3쌍 미만인 담당 게임이 남아 있고 통과 full FLASH 네 개가 아직 확보되지 않았다면,
그 게임이 이미 다른 GPU에서 실행 중이어도 다른 시드를 배정받습니다. 모두 3쌍 이상이고
네 번째 쌍 작업이 이미 실행 중이거나, 필요한 baseline 자리가 모두 확보되어 추가 작업을
배정할 수 없으면 학습을 시작하지 않고 바로 종료합니다. 대기하거나 나중에 자동으로
실행하지 않습니다.
해당 서버가 모든 목표를 이미 달성했으면 진행 상황을
표시하고 종료합니다. 중단된 작업이 있다면 그 작업을 재개할 수 있습니다.
이 실행은 Gopher·Krull을 포함한 서버의 담당 목록에서 작업하며 담당 게임 수는
바꾸지 않습니다. 모든 담당 게임은 다른 GPU에서 실행 중이어도 3쌍이 부족하면
다른 시드를 배정받을 수 있습니다. 별도의 `N_once_train.sh`는 필요하지 않습니다.

## shared warmup이 끝나지 않는 예외

50,000 결정에 도달해도 진행 중인 에피소드를 강제로 끊지 않습니다. 각 환경의
에피소드가 끝나면 차례로 멈추고, 모든 환경이 멈춘 뒤 shared warmup을 저장합니다.

학습 한도를 다 쓸 때까지 경계를 만들지 못하면, 해당 시드의 full FLASH를
처음부터 다시 학습합니다. 이 결과가 위의 두 조건 중 하나를 만족할 때만 baseline 점수를
구합니다. warmup이 환경 일시 정지·수집 재개 없이 전체 학습 예산을 채웠다면,
이미 retrieval OFF로 학습한 모델이므로 마지막 체크포인트를 100회 최종 평가합니다.
baseline을 다시 학습하지 않습니다. 조건을 확인할 수 없거나 warmup이 미완료이면
baseline도 동일한 시드로 처음부터 학습합니다.
재개한 warmup이 이미 목표 결정을 넘겼는데 에뮬레이터 상태가 없는 경우도
이 경로를 사용합니다. 환경 초기화를 자연스러운 에피소드 종료로 취급하지 않습니다.

추가 시드로 세지 않으며 상태의 `warmup_fallback`에 처리 방식을 기록합니다.
W&B, 완료/성공 통계와 상위 4개 보관 규칙은 동일하게 적용합니다. 독립 학습 도중
중단된 분기는 다음 실행에서 처음부터 다시 시작하고, 이전 파일은 trial 안에
보관합니다. 완료된 평가 결과는 재사용합니다. CUDA 오류 등 다른 실패는 이
예외 처리로 숨기지 않고 worker를 종료합니다.
재사용 baseline은 평가가 중단돼도 학습을 다시 하지 않습니다. 원본 체크포인트를
baseline 디렉터리에 hard link하고, `training_source.json`에 출처를 기록합니다.
평가 시 optimizer·replay를 복원하거나 체크포인트를 다시 저장하지 않습니다.

50,000 이전 후보를 매 에피소드마다 저장하는 방식은 추가하지 않았습니다.
[실측 비용과 판단 근거](warmup_fallback_cost.md)를 참고하세요. 새 예외 처리는
재시작한 worker부터 적용되며, 실행 중인 worker를 자동으로 중단하지 않습니다.

## 하루 한 번: 서버마다 명령 하나

```bash
python reproduction_cluster.py sync
```

자기 서버의 결과 요약만 커밋하고, `git pull --rebase`와 `git push`를 실행합니다.
다른 서버의 결과가 들어오면 서버별 라운드, 완료 학습 수, 성공 쌍 수와 아직
시작하지 않은 게임 목록을 표시합니다. 다른 서버가 나중에 push한 결과는 다음
동기화에서 반영됩니다. 학습을 멈추거나 파라미터를 업로드하지 않습니다.
표의 Round는 해당 서버에서 시작한 최대 차수이며, 각 게임의 진행 차수는
다를 수 있습니다. 라운드 전체의 완료를 기다리는 동기화 지점은 없습니다.

Git 인증, 사용자 이름/이메일, 브랜치 upstream은 먼저 설정되어 있어야 합니다.
관련 없는 tracked 파일에 수정이 있으면 동기화를 중단하고 해당 파일을 알려줍니다.
기존 수정 사항을 자동으로 stash/reset하거나 함께 커밋하지 않습니다.
동시 push 때문에 거절되면 최대 세 번 pull/push를 재시도합니다. 충돌이나 인증
오류는 Git 메시지에 따라 해결한 뒤 같은 sync 명령을 다시 실행하세요. 로컬
학습 결과와 생성된 커밋은 보존됩니다.

```bash
# Git에 접근하지 않고 현재 서버의 요약만 생성
python reproduction_cluster.py sync --local-only

# 마지막으로 교환한 세 서버의 요약 보기
python reproduction_cluster.py status
```

`reports/`에는 점수, 시드, 상태, 집계만 기록합니다. 실행 중 상태 파일,
Python 경로, 파라미터 경로, replay 파일, W&B 인증 정보는 업로드하지 않습니다.

## 속도 차이가 클 때만: 미시작 게임 넘기기

예를 들어 C의 Road Runner가 아직 시작되지 않았고 A가 빠르다면 **C에서**:

```bash
python reproduction_cluster.py move road_runner a
python reproduction_cluster.py sync
```

이후 A에서 평소의 sync를 실행하면 다음 작업 배정부터 Road Runner가 포함됩니다.
C는 move 직후 로컬 큐에서 그 게임을 제외하므로, Git을 공유하는 동안 중복
학습하지 않습니다. 받은 서버가 이미 다음 라운드에 있어도 이 게임의 첫 시드는
3090부터 시작합니다. 현재 실행 중인 worker는 최신 코드로 재시작해야 이 규칙을
적용받습니다.

일반 `move`는 이미 시작한 게임의 이동을 거절합니다. 해당 게임의 shared replay가 절대 경로로
참조되고, 상위 4개 산출물이 원래 서버에 있기 때문입니다. 이 경우 각 서버는
자기 게임의 다음 라운드를 계속 진행하면 됩니다. 이미 시작한 B/C 게임의
배정 변경은 위의 `handoff` 절차를 사용합니다. 공유 게임 Gopher·Krull은 이미 모든
서버에서 실행하므로 `move`를 허용하지 않습니다.

서버 B/C로 이 저장소를 옮길 때는 Git에 포함된 파일만 옮기세요. 다른 서버의
`reproduction.local.json`이나 `results/reproduction/`을 복사하면 로컬 실행 상태가
섞입니다. 기존 모델을 그대로 보관하려면 `callbacks/`를 원래 서버에 유지합니다.
