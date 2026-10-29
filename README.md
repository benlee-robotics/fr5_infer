# fr5_infer

`fr5_vla`에서 학습한 LeRobot 정책 체크포인트(Pi0.5)를 FR5 실기에서 실행하는 도구다.
조작은 웹 페이지에서 한다.

`fr5_teleop/docs/policy_deploy.md`의 설계 원칙을 따르되, worker에 통합하지 않고 별도
프로젝트로 만들었다. 대신 `fr5_teleop`의 `teleop` 패키지를 라이브러리로 import해서
현장 검증이 끝난 코드를 그대로 쓴다. fr5_teleop 코드는 수정하지 않는다.

| 가져다 쓰는 것 | 위치 |
|---|---|
| 관측 벡터 변환 (`pose_vector`, rot6d, 쿼터니언) | `teleop/recording/lerobot_convert.py` |
| 안전 제한 (`limit_home_step`: 스텝·각스텝·workspace), 홈 복귀 궤적, slerp | `teleop/control_math.py` |
| 로봇 SDK 클라이언트, fake 로봇 | `teleop/robot/fairino_client.py`, `fake_client.py` |
| RealSense / fake 카메라 | `teleop/recording/camera.py` |
| 로봇 IP, workspace, 속도, 그리퍼, 홈 자세, 카메라 serial | `fr5_teleop/config.yaml` |

## 구조

```text
메인 프로세스 ── aiohttp 웹 (127.0.0.1:8080)
              └─ policy 스레드: 관측 → 추론(GPU) → 시각 정렬 → 블렌딩 → 그리퍼 판정 → mailbox
Robot 프로세스 ── 8 ms 루프: mailbox 궤적 샘플링 → 안전 제한 → ServoCart, 그리퍼 시퀀스
Camera 프로세스 ×2 (top, wrist) ── 최신 프레임 1장을 공유 메모리에 덮어쓰기
```

- 로봇 루프는 추론을 기다리지 않는다. mailbox를 읽기만 한다.
- 모든 시각은 `time.monotonic_ns()`이다. 청크의 k번째 행에 `관측 시각 + k/fps`를 붙이고,
  추론하는 동안 지나간 점은 버린다. 추론 지연은 이 단계에서 보정된다.
- 궤적이 끝나면(마지막 점 + `trajectory_grace_s`) 마지막 목표를 유지하지 않고 서보를 멈춘다.
- 그리퍼가 바뀌는 시점에서 궤적을 자른다. 그 시점에 도달하면 다음 순서로 동작한다
  (fr5_teleop과 같은 순서).
  1. 서보 정지
  2. `MoveGripper`
  3. 완료 대기
  4. 그리퍼 완료 **이후에 찍은 관측**으로 만든 궤적이 들어오면 재개
- 다음 경우에는 정책 시작을 거부한다.
  - 첫 목표가 현재 TCP에서 `max_start_offset_mm` / `max_start_offset_deg`보다 멀 때
  - 현재 TCP가 workspace 밖일 때

## 버전 맞추기 (가장 중요)

정책은 **학습 PC와 같은 lerobot·transformers 버전**으로 로드해야 한다. 학습 PC 환경은
lerobot 0.6.2(소스 체크아웃 `afbf3f14`, 2026-10-20) + transformers 5.5.4이고, 이 버전은
Python 3.12가 필요하다. `requirements.txt`가 이 조합을 고정하고, `requirements-lock.txt`가
전체 의존성을 고정한다.

버전이 다르면 체크포인트를 읽지 못하거나, 읽어도 예측이 달라진다. 2500 스텝 체크포인트로
비교한 결과는 다음과 같다(같은 데이터셋 50샘플, `scripts/eval_offline.py`).

| | lerobot 0.4.4 (필드 제거 우회) | lerobot 0.6.2 (학습 PC와 동일) |
|---|---|---|
| 평균 위치 오차 | 56.1 mm | 35.9 mm |
| 회전 오차 | 4.14° | 2.67° |
| 그리퍼 일치율 | 0.869 | 0.976 |

학습 PC에서 lerobot을 업데이트하면 `requirements.txt`의 커밋을 바꾸고 lock을 다시 만든다.

## 설치 (개발 PC, conda env `fr5-infer`)

```bash
conda create -n fr5-infer python=3.12
conda activate fr5-infer
cd ~/project/fr5_ws/fr5_infer
pip install -r requirements-lock.txt pytest==8.4.1
```

- Docker 이미지와 같은 lock 파일을 쓰므로 버전이 이미지와 같다.
- `teleop` 패키지는 같은 워크스페이스의 `../fr5_teleop`에서 자동으로 찾는다. 다른 위치에
  있으면 `FR5_TELEOP_ROOT`로 지정한다.
- `lerobot[phone]` extra는 설치하지 않는다. PyPI의 `teleop` 패키지가 함께 깔려서
  fr5_teleop의 `teleop`과 이름이 충돌한다.

체크포인트는 학습 PC의 `pretrained_model` 폴더를 통째로
`checkpoints/fr5_pick_cube_pi05_expert/<스텝>/pretrained_model/`에 복사한다(~9 GB).
`training_state/`는 필요 없다.

## 오프라인 평가 (로봇 없이)

새 체크포인트는 실기 전에 녹화 데이터로 먼저 확인한다. 녹화된 관측을 로봇과 같은 추론
코드에 넣고, 예측 청크를 실제 action과 비교한다. "가만히 있기"(현재 state 유지)를 기준선으로
함께 출력한다.

```bash
python scripts/eval_offline.py \
    --checkpoint checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model \
    --dataset ../fr5_teleop/datasets/fr5_pick_cube_20261027 --episodes 0 40 80
```

2500 스텝 체크포인트의 결과는 다음과 같다(학습에 쓴 에피소드라서, 일반화가 아니라
학습 상태를 보는 값이다).

- 그리퍼 일치율 0.976
- 움직이는 구간: 위치 오차 39.9 mm (가만히 있기 75.7 mm)
- 첫 스텝 오차: p50 16 mm, p90 32 mm. 데이터셋에서 명령과 측정 사이 차이는 평균 4.9 mm다.
- 50샘플 중 20개가 첫 스텝에서 20 mm를 넘는다. 기본 `max_start_offset_mm: 20`에서는
  시작이 자주 거부될 수 있다.

## 청크 이음매: RTC와 시간 앙상블

기본(v0.1.0과 같음)으로는 청크마다 새 노이즈에서 샘플링한다. 그래서 0.67 s마다 경로가 바뀌고,
그 차이를 `blend_s` 0.15 s 안에서 흡수한다. 두 기법을 켤 수 있다.

- **RTC** (`policy.rtc.enabled`, pi05 전용): lerobot 0.6의 추론 시점 guided RTC다. 재학습이 필요
  없다. 이전 청크에서 아직 실행하지 않은 부분(모델 공간)을 넘긴다. 추론 동안 실행될 스텝
  (`지연 EMA + min_lead_s`)은 이전 청크에 고정되고, `execution_horizon` 스텝까지 가중치가 1에서 0으로
  줄어든다. 그리퍼 동작 뒤와 에피소드 시작에는 이어 붙이지 않는다.
- **시간 앙상블** (`execution.ensemble`): ACT 방식이다. 시각마다 그 시각을 예측한 모든 청크의 가중
  평균을 낸다. i번째 예측(0 = 가장 오래된 것)의 가중치는 `exp(-ensemble_decay·i)`다. 그리퍼 출력도
  평균한다.

녹화 에피소드를 로봇처럼 순차 재생해 이음매를 비교한다.

```bash
python scripts/eval_chunk_continuity.py \
    --checkpoint checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model \
    --dataset ../fr5_teleop/datasets/fr5_pick_cube_20261027 --episodes 0 40 80
```

002500, 에피소드 0/40/80, replan 20, 지연 5스텝, RTX 5080 결과다. "튐"은 청크를 바꾸는 시점에
이전 궤적과 새 궤적 사이의 거리이고, "오차"는 실행 구간의 녹화 action 대비 위치 오차다.

| 방식 | 튐 p50 / p90 / max (mm) | 회전 튐 max (°) | 오차 (mm) | 지연 p50 (ms) |
|---|---|---|---|---|
| 독립 샘플 (기본) | 35.2 / 66.4 / 155.8 | 11.1 | 28.4 | 126 |
| 앙상블 | 11.1 / 24.5 / 40.7 | 5.5 | 32.2 | 126 |
| **RTC (horizon 10)** | **3.4 / 7.0 / 12.9** | **0.95** | **26.6** | 155 |
| RTC (horizon 20) | 1.6 / 3.2 / 3.9 | 0.36 | 34.6 | 152 |
| RTC + 앙상블 | 6.3 / 16.4 / 31.0 | 1.9 | 32.0 | 155 |

- 독립 샘플의 튐 p50 35 mm를 0.15 s에 흡수하려면 약 230 mm/s가 필요하다. 그래서 속도 제한에 걸린다.
  2026-10-29 현장에서 100 mm/s일 때 생긴 jerking과 맞는 값이다.
- RTC(horizon 10)는 튐을 1/10로 줄이면서 오차는 늘리지 않는다. horizon 20은 더 매끄럽지만 이전 계획을
  오래 따라가 오차가 커진다.
- 앙상블은 튐을 줄이지만 반응이 느려져 오차가 늘고, RTC와 같이 쓰면 오히려 나빠진다.
- RTC는 denoising 스텝마다 역전파를 한 번 더 해서 지연이 약 30 ms 늘어난다. GPU 메모리는 거의 같다
  (9.76 GB).

## Docker (현장 배포)

현장 PC에서는 conda env 대신 이미지로 실행한다. 이미지 안에는 다음이 들어 있다.

- `requirements-lock.txt`로 고정한 의존성: Python 3.12, torch 2.7.1+cu128,
  lerobot 0.6.2(학습 PC와 같은 커밋), transformers 5.5.4, pyrealsense2
- fr5_teleop 이미지(`TELEOP_IMAGE`, 기본 `0.4.3`)에서 복사한 `teleop` 패키지, `web/`, Fairino SDK

안전 제한 코드는 현장 텔레옵 이미지와 같은 버전이다. 체크포인트는 이미지에 넣지 않고
마운트한다.

### 현장 PC 준비 (한 번)

호스트에는 NVIDIA 드라이버와 **nvidia-container-toolkit**이 필요하다. CUDA 툴킷은
필요 없다. 드라이버는 570 이상이어야 cu128 휠이 돈다.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi        # GPU가 보이면 성공
```

디렉터리 배치는 다음과 같다. compose가 `../fr5_teleop`을 마운트하므로, fr5_infer는
fr5_teleop 옆에 둔다.

```text
~/fr5_teleop/config.yaml            현장 config (그리퍼, 카메라 serial, 홈)
~/fr5_teleop/datasets/<학습 데이터셋>/fr5_conversion.json, meta/info.json
~/fr5_infer/checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model/
```

- pi05에는 데이터셋 중 `fr5_conversion.json`과 `meta/info.json`만 있으면 된다.
- `--backend replay`에는 `data/` parquet도 필요하다.
- 체크포인트 안에 `tokenizer/`가 있으면 그것을 쓴다. 이 경우 HuggingFace 로그인이나
  캐시가 없어도 된다.

### 이미지 배포 (ghcr)

fr5_teleop과 같은 방식이다. `v*` 태그를 push하면 GitHub Actions(`.github/workflows/docker.yml`)가
순서대로 처리한다.

1. 태그와 `fr5_infer.__version__`이 같은지 확인
2. 이미지 안에서 pytest 실행
3. `ghcr.io/benlee-robotics/fr5_infer:<버전>`, `:latest`로 push

```bash
# 개발 PC: fr5_infer/__init__.py의 __version__을 올리고 커밋한 뒤
git tag v0.1.2 && git push origin main v0.1.2
```

- 이미지는 공개다(로그인 불필요). 처음 push된 패키지는 GitHub에서 private로 만들어지므로,
  패키지 설정(Package settings → Change visibility)에서 한 번 public으로 바꾼다.
- 현장 PC에서는 빌드하지 않고 받기만 한다.

  ```bash
  echo "INFER_TAG=0.1.2" > .env
  docker compose pull
  ```

- 로컬에서 직접 빌드하거나 테스트하려면 다음을 쓴다.

  ```bash
  docker compose build                       # 인터넷 필요, 처음엔 오래 걸림(이미지 ~12.5 GB)
  docker build --target test .               # 이미지 안에서 pytest
  ```

### 실행

`runs/`와 `checkpoints/`는 먼저 만들어 둔다. 없는 폴더는 Docker가 root 소유로 만들고,
그러면 컨테이너 사용자(UID 1000)가 기록을 쓰지 못한다.

```bash
mkdir -p runs checkpoints
docker compose run --rm infer --mode fake --backend replay
docker compose run --rm infer --mode shadow
docker compose run --rm infer --mode real --confirm-hardware
docker compose run --rm --entrypoint python infer -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
docker compose run --rm --entrypoint python infer scripts/eval_offline.py \
  --checkpoint checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model \
  --dataset ../fr5_teleop/datasets/fr5_pick_cube_20261027 --episodes 0 40 80
```

- 웹 페이지는 현장 PC 브라우저에서 `http://127.0.0.1:8080/`로 연다.
- 종료는 Ctrl+C로 한다. 서보 정지 → 연결 종료 순서로 정상 종료된다.
- 의존성을 바꿀 때는 `requirements.txt`를 고친 뒤
  `docker build --target lock --output type=local,dest=. .`로 lock 파일을 다시 만든다.

## 실행 (conda env)

```bash
python -m fr5_infer --mode fake   --backend replay            # 하드웨어·모델 없이 전체 경로 확인
python -m fr5_infer --mode fake   --backend pi05              # 모델 로드·지연·VRAM 확인
python -m fr5_infer --mode shadow --backend pi05              # 실기 연결, 로봇은 움직이지 않음
python -m fr5_infer --mode real   --backend replay --confirm-hardware
python -m fr5_infer --mode real   --backend pi05   --confirm-hardware
```

브라우저에서 `http://127.0.0.1:8080/`을 연다.

- **버튼**
  - 홈으로: `home.tcp`로 복귀
  - 정책 시작
  - 정지 (Esc 키도 정지)
  - fault 리셋
  - 지시문 적용
- **화면 표시**
  - 로봇 상태와 사유, TCP, 명령 TCP, 그리퍼
  - 추론 지연(EMA 포함), 보낸 점/버린 점 수, 남은 궤적
  - 카메라 미리보기 2개

| 모드 | 로봇 | 카메라 | 용도 |
|---|---|---|---|
| `fake` | FakeRobotClient (홈 자세에서 시작) | 합성 프레임 | 파이프라인·UI 확인 |
| `shadow` | 실기 연결. `initialize`, 서보, 그리퍼 명령을 보내지 않음 | RealSense | 관측·지연·출력 확인. "보냈을 명령"이 `ticks.npy`에 남음 |
| `real` | 실기 실행 | RealSense | 배포 |

`--backend replay`는 데이터셋 에피소드(`--episode N`)의 action을 경과 시간에 맞춰 50스텝
청크로 돌려준다. 추론 지연은 0.1 s로 흉내 낸다. 관측은 쓰지 않으므로, 그 에피소드를
시작했던 자세(홈)에서 시작해야 한다.

## 실기 절차

1. **fr5_teleop 컨테이너를 멈춘다** (`docker stop fr5`). 로봇에 명령을 보내는 프로그램은
   하나만 떠 있어야 한다.
2. `fr5_teleop/config.yaml`을 확인한다.
   - `gripper.enabled: true` (real 모드는 이 값이 없으면 시작을 거부한다)
   - `recording.cameras`의 `top`, `wrist` serial
   - `home`
3. 작업자는 비상정지 버튼 옆에 있는다. 로봇 주변을 비운다.
4. 다음 순서로 진행한다.
   1. `shadow`: 홈 자세에서 state가 가이드의 홈 벡터와 같은지, 지연이 `replan_steps`
      (0.67 s)보다 충분히 짧은지 확인한다.
   2. `real` + `replay`: 녹화된 궤적으로 서보·그리퍼 시퀀스를 확인한다.
   3. `real` + `pi05`.
5. 처음에는 `safety.max_velocity_mm_s`를 50으로 낮춰 시작한다.

### 속도 제한과 학습 분포

실제 제한은 `min(fr5_teleop motion/orientation, safety)` 값이다. 기본값은 100 mm/s,
15 deg/s다. 시연 데이터(`fr5_pick_cube_20261027`)의 action 속도는 다음과 같다.

| 지표 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 위치 (mm/s) | 72 | 178 | 245 | 329 |
| 회전 (deg/s) | 2.9 | 14.4 | 18.0 | — |

100 mm/s 제한에서는 프레임의 약 37%가 제한에 걸려 로봇이 시연보다 느리게 따라간다.
안전을 위해 이렇게 시작하지만, 그만큼 정책이 보는 상태는 학습 분포에서 벗어난다.
동작이 확인되면 `max_velocity_mm_s`를 fr5_teleop 값(250)까지 단계적으로 올린다.

## 설정 (`config.yaml`)

| 항목 | 의미 |
|---|---|
| `policy.conversion` | 학습 데이터셋의 `fr5_conversion.json`. fps, rotation, 카메라, 카메라 지연을 읽는다. 시작할 때 체크포인트 `config.json`의 feature와 대조하고, 다르면 시작을 거부한다 |
| `execution.replan_steps` | 마지막 관측 이후 이만큼 스텝이 지나면 다시 추론한다(가이드 권장 15~25) |
| `execution.refill_margin_s` | 남은 궤적이 `지연 EMA + 이 값`보다 짧아도 다시 추론한다 |
| `execution.blend_s` | 새 청크의 앞부분을 이전 궤적과 코사인 가중으로 섞는 구간 |
| `execution.ensemble`, `ensemble_decay` | ACT식 시간 앙상블과 가중치 감쇠 (위 "청크 이음매") |
| `policy.rtc.*` | RTC 사용 여부, `execution_horizon`, `max_guidance_weight`, `schedule` (위 "청크 이음매") |
| `execution.lookahead_s` | 서보 지연 보상. 궤적을 이만큼 앞에서 샘플링한다 |
| `safety.max_orientation_deviation_deg` | 홈 자세 기준 절대 자세 범위 |
| `safety.max_episode_s` | 에피소드 최대 시간 |
| `gripper.*` | 히스테리시스(닫힘 > 0.6, 열림 < 0.4)와 최소 유지 시간. `settle_s`, `max_settle_s`: 그리퍼 완료 뒤 측정 위치가 `settle_s` 동안 그대로일 때(최대 `max_settle_s`) 서보를 다시 시작한다. 움직이는 동안 측정 위치는 0으로 읽히고 완료 0.1~0.3 s 뒤에 최종값이 된다 |

## 기록 (`runs/<시각>_<모드>/`)

| 파일 | 내용 |
|---|---|
| `meta.json` | 모드, 백엔드·체크포인트, 변환 기록, 지시문, 설정, 종료 사유, 추론 지연 통계 |
| `inferences.npz` | 추론마다 관측 시각, 지연, `observation.state`, 원본 청크(T×10), 보낸/버린 점 수, 그리퍼 전환 시각, RTC prefix 길이와 고정 스텝 수(`rtc_prefix_steps`, `rtc_delay_steps`, 0이면 guidance 없음) |
| `ticks.npy` | 8 ms tick마다 측정 TCP, 그리퍼, 보간 목표(`desired`), 제한 후 명령(`cmd`). dtype은 `robot_process.TICK_DTYPE` |
| `robot_end_reason.txt` | 로봇 쪽 종료 사유 |
| `images/` | `log.save_images: true`일 때 추론 입력 프레임 |

## 테스트

```bash
python -m pytest -q
```

- 변환
  - 가이드의 홈 TCP·rot6d와 일치
  - 데이터셋 첫 프레임과 일치
  - rot6d 역변환 왕복
- 궤적
  - 시각 정렬, 블렌딩 연속성
  - 히스테리시스·dwell
  - 시간 앙상블 평균·감쇠·격자 밖 보간·초기화
- RTC
  - 설정 검증, prefix 시작 위치·지연 스텝, 겹침 없음·연결 끊김
- 로봇 루프 (FakeRobotClient)
  - 시작 거리 거부, 시작 전에 쓰인 궤적 무시
  - 궤적 소진 시 정지, 속도 제한
  - 그리퍼 정지 → 재개, 완료 뒤 위치 안정 대기와 상한, shadow 무명령, 정지 명령

## 한계

- 데드맨 스위치가 없다. 정지는 웹의 정지 버튼(Esc)과 하드웨어 비상정지로 한다.
- 단팔만 지원한다 (`robot.arms` 양팔 설정은 거부).
- 컨테이너 안에서 GPU를 쓰는 것은 개발 PC에 nvidia-container-toolkit이 없어 확인하지 못했다.
  pi05 추론은 같은 lock 파일로 만든 conda env에서 확인했다.
- 충돌 검사는 없다. workspace 박스, 속도 제한, 자세 범위로 대신한다.
# fr5_infer
