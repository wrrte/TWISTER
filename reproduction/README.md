# GPU 4개 / 4개 / 1개를 세 서버에서 사용하기

`N_train.sh`는 GPU별로 `reproduce.py` worker를 실행합니다. 각 서버의 worker는
로컬 `state.json`을 공유합니다. 서버 간에는 `servers.json`의 게임 배정과
`reports/a.json`, `b.json`, `c.json`의 작은 결과 요약만 Git으로 교환합니다.
`job_queue_3090.txt`와 다른 `job_queue_*.txt`는 이 실행에서 사용하지 않습니다.

배정은 A(4 GPU) 11게임, B(4 GPU) 12게임, C(1 GPU) 3게임입니다.
서버 A는 RTX 3090 서버, B는 RTX A6000 서버, C는 나머지 1-GPU 서버입니다.
게임 배정은 이 역할 등록으로 결정하며, 학습 시작 시 선택한 GPU 모델과
PyTorch의 CUDA 인식도 검사합니다. 시도 횟수가 적은 게임부터 우선 배정하며,
같은 횟수이면 담당 게임 목록의 순서를 따릅니다. 다른 게임이 아직 학습 중이어도
이미 끝난 게임은 다음 시드를 실행할 수 있습니다. 단, 같은 게임의 retrieval와
조건부 baseline이 끝나기 전에는 그 게임의 다음 시드를 동시에 실행하지 않습니다.
게임별 시드는 `3090, 3100, 3110, ...`입니다. 성공 시드 4개 확보,
baseline 조건, full FLASH 점수 상위 4개 보관 규칙은 그대로 적용합니다.

## 최초 한 번

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
새 worker는 배포된 최신 코드를 사용합니다. 이미 실행 중인 worker는 이번 작업을
끝낸 뒤 재시작하면 새 코드가 적용되며, 저장된 warmup/평가는 재사용합니다.
라운드 완료 대기를 제거한 스케줄러를 적용할 때는 모든 worker를 최신 코드로
재시작해야 합니다. 기존 상태와 파라미터를 계속 사용하며 초기화는 필요하지 않습니다.

## 남는 GPU에서 한 번만 돕기

예를 들어 같은 서버의 GPU 5를 잠시 사용할 수 있다면:

```bash
./5_train.sh --once
```

기존 worker들과 같은 상태 파일에서 다음 작업 한 건을 중복 없이 가져옵니다.
shared warmup → full FLASH 학습/최종 평가 → 논문 HNS − 0.1 이상일 때 같은
warmup에서 baseline 실행까지 마친 뒤 종료합니다. 기준 미달이면 baseline을
생략하고 종료합니다. 결과는 정규 실행과 같은 완료/성공 통계, W&B 설정,
게임별 상위 4개 보관 규칙에 포함되며 다음 sync에서 공유됩니다.

예를 들어 11개 게임 중 6개 시도가 완료됐고 5개가 다른 GPU에서 진행 중이라면,
완료된 게임 중 성공 시드 4개가 아직 없는 게임의 다음 시드를 실행합니다.
성공 시드가 부족한 모든 게임이 이미 다른 GPU에서 진행 중일 때만 학습을
시작하지 않고 바로 종료합니다. 대기하거나 나중에 자동으로 실행하지 않습니다.
해당 서버가 모든 목표를 이미 달성했으면 진행 상황을
표시하고 종료합니다. 중단된 작업이 있다면 그 작업을 재개할 수 있습니다.
이 실행은 서버의 기존 담당 게임 안에서만 작업하며 담당 게임 수는 바꾸지
않습니다. 별도의 `N_once_train.sh`는 필요하지 않습니다.

## shared warmup이 끝나지 않는 예외

50,000 결정에 도달해도 진행 중인 에피소드를 강제로 끊지 않습니다. 각 환경의
에피소드가 끝나면 차례로 멈추고, 모든 환경이 멈춘 뒤 shared warmup을 저장합니다.

학습 한도를 다 쓸 때까지 경계를 만들지 못하면, 해당 시드의 full FLASH를
처음부터 다시 학습합니다. 이 결과가 논문 HNS − 0.1 이상일 때만 baseline 점수를
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

이미 시작한 게임은 이동을 거절합니다. 해당 게임의 shared replay가 절대 경로로
참조되고, 상위 4개 산출물이 원래 서버에 있기 때문입니다. 이 경우 각 서버는
자기 게임의 다음 라운드를 계속 진행하면 됩니다. `servers.json`을 직접 편집해
실행 중인 게임의 담당 서버를 바꾸지 마세요.

서버 B/C로 이 저장소를 옮길 때는 Git에 포함된 파일만 옮기세요. 다른 서버의
`reproduction.local.json`이나 `results/reproduction/`을 복사하면 로컬 실행 상태가
섞입니다. 기존 모델을 그대로 보관하려면 `callbacks/`를 원래 서버에 유지합니다.
