# 현장 검증 작업 지시서 (fr5_infer 0.1.0, Pi0.5 체크포인트 002500)

이 문서는 현장 PC에서 작업하는 Claude에게 주는 지시서다. 작업자(사람)와 함께
아래 순서대로 검증하고, 결과를 보고서로 남긴다. 작업자와는 한국어로 대화한다.

## 0. 배경

- **프로젝트:** fr5_infer는 텔레옵 시연으로 학습한 LeRobot Pi0.5 정책을 FR5 실기에서
  실행한다. 조작은 웹 페이지(`http://127.0.0.1:8080/`)에서 한다. 구조와 설정은
  `README.md`를 본다.
- **fr5_teleop과의 관계:** 별도 프로그램이지만, 로봇 클라이언트·안전 제한·관측 변환은
  fr5_teleop v0.4.1 코드를 그대로 쓴다(이미지에 포함). 로봇 IP, 그리퍼, 홈 자세,
  workspace, 카메라 serial은 **`~/fr5_teleop/config.yaml`(현장 config)** 에서 읽는다.
- **배포 방식:** Docker 이미지 `ghcr.io/benlee-robotics/fr5_infer:0.1.0`. ghcr에는 올라가 있지
  않으므로 USB로 받은 파일을 `docker load`한다. 컨테이너는 `compose.yaml`로 실행한다.
- **현장 PC:** RTX 4090(24 GB), NVIDIA 드라이버 580 계열(작업자 기억, 3-1에서 확인).
- **로봇:** FR5, IP `192.168.57.3`, 그리퍼 사용, 카메라 `wrist`(D435i)·`top`(D435).
- **정책:** `fr5_pick_cube_pi05_expert`의 **2500 스텝 중간 체크포인트**(전체 10000).
  - 학습 데이터셋: `fr5_pick_cube_20261027`
  - 지시문: `pick up the cube and place it in the box`
  - 30 Hz, 50스텝(1.67 s) 청크

### 이번 검증의 목적

1. 현장 PC에서 이미지가 GPU로 정책을 돌리는지 확인한다(개발 PC에서는 확인 못 함).
2. 실제 카메라·로봇 상태로 관측과 추론이 정상인지 확인한다(shadow, 로봇 정지).
3. 저속으로 실기 실행 경로를 확인한다. 순서는 녹화 재생(replay) → 정책(pi05).

**2500 스텝 체크포인트로 과제 성공을 기대하지 않는다.** 실기 단계의 목적은 안전 동작
확인과 정책 행동의 정성적 관찰이다.

### 개발 PC 결과 (비교 기준)

오프라인 평가는 학습 에피소드 0/40/80, 50샘플, RTX 5080에서 했다.

| 항목 | 값 |
|---|---|
| 그리퍼 일치율 | 0.976 |
| 평균 위치 오차 | 35.9 mm (가만히 있기 58.5) |
| 움직이는 구간 위치 오차 | 39.9 mm (가만히 있기 75.7) |
| 회전 오차 | 2.67° |
| 첫 스텝 위치 오차 | p50 16 mm, p90 32 mm. 50개 중 20개가 20 mm 초과 |
| 추론 지연 | 평가 p50 179 ms, 앱 EMA 약 160 ms |
| GPU 메모리 | 9.7 GB |

### 이미 알려진 사항 (문제로 보고하지 않는다)

- 모델 로드 때 `Vision embedding key might need handling: ...patch_embedding...` 경고가
  2줄 나온다. lerobot이 가중치를 변환하며 항상 찍는 로그이고, `All keys loaded successfully!`
  가 나오면 정상이다.
- 오프라인 평가에서 `'torchcodec' is installed but cannot be loaded ... Falling back to 'pyav'`
  가 나온다. 영상 디코딩 경로만 바뀌는 것이고 결과에는 영향이 없다.
- **첫 스텝 오차가 커서 `start_offset_XXmm_YYdeg`로 시작이 거부될 수 있다**(기본 한도
  20 mm, 5°). 3-7에서 다루는 예상된 현상이다.
- 기본 속도 제한 100 mm/s는 시연(p99 245 mm/s)보다 느리다. 로봇이 시연보다 천천히
  따라가는 것은 설계다.
- shadow 모드에서는 홈 복귀가 거부된다(`home_refused_shadow_mode`). 설계다.
- 정책 실행 중 그리퍼가 움직일 때 서보가 멈췄다 다시 시작하는 것은 설계다(FR5는 서보 중
  `MoveGripper`를 거부). 상태가 `GRIPPER` → `STARTING` → `RUNNING`으로 바뀐다.
- 궤적이 끝나면 로봇은 마지막 자세를 유지하지 않고 서보를 멈춘다(`trajectory_exhausted`).
  설계다.

### USB로 전달되는 것

| 파일 | 놓을 위치 | 크기 |
|---|---|---|
| `fr5_infer/` (소스, compose, config, scripts, 이 문서) | `~/fr5_infer/` | 작음 |
| `fr5_infer_0.1.0.tar.gz` (이미지) | 아무 곳, `docker load` 후 삭제 가능 | ~4.4 GB |
| `checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model/` | `~/fr5_infer/checkpoints/...` 같은 경로 | 8.8 GB |
| `fr5_pick_cube_20261027/` (데이터셋) | `~/fr5_teleop/datasets/fr5_pick_cube_20261027/` | 390 MB |

## 1. 반드시 지킬 규칙

1. **로봇이 움직일 수 있는 실행(`--mode real`)은 매번 작업자에게 먼저 알리고 동의를 받은
   뒤 실행한다.** 작업자가 비상정지 버튼 옆에 있는지 확인한다. 웹의 **정지** 버튼(또는
   Esc)도 작업자가 누를 수 있게 페이지를 열어 둔다.
2. **정책이 로봇을 움직이는 것은 이번이 처음이다.** 3-7 시작 전에 작업자에게 다음을 알리고
   확인받는다.
   - 2500 스텝 중간 체크포인트이고 동작이 엉뚱할 수 있다.
   - 로봇 주변(특히 아래쪽, 박스 주변)에 사람과 깨질 물건이 없어야 한다.
   - 첫 실행은 50 mm/s다.
3. **fr5_teleop 컨테이너와 동시에 실행하지 않는다.** 로봇에 명령을 보내는 프로그램은
   하나만 떠 있어야 한다. 실행 전마다 `docker ps`로 `fr5`가 없는지 확인한다.
4. Claude는 로봇과 화면을 직접 볼 수 없다. 움직임 방향, 부드러움, 그리퍼 동작, 웹 화면
   표시는 작업자에게 구체적으로 물어보고 답을 그대로 기록한다. 추측으로 "통과"라고 적지
   않는다.
5. **`~/fr5_infer/fr5_infer/` 코드와 이미지는 수정하지 않는다.** 문제를 찾으면 증상, 로그,
   재현 방법을 보고서에 적는다. `~/fr5_infer/config.yaml`과 `~/fr5_teleop/config.yaml`은
   이 문서가 지시하거나 작업자가 요청할 때만 고치고, 바꾼 값과 이유를 보고서에 적는다.
6. 이상 동작(급가속, 반대 방향, 크게 도는 회전, 멈추지 않음, 테이블·박스로 파고듦)이
   보이면 다음 순서로 대응한다.
   1. 즉시 작업자에게 정지를 요청한다(웹 정지 / Esc).
   2. Claude는 `curl -s -XPOST 127.0.0.1:8080/api/stop`을 보낸다.
   3. 그래도 멈추지 않으면 `docker stop fr5_infer`.

   비상정지는 작업자가 판단한다.
7. 커밋과 push는 작업자가 요청할 때만 한다. 체크포인트, 데이터셋, `runs/`는 git에 넣지
   않는다.

## 2. 실행 방식 (Claude용)

Claude의 셸은 대화형 TTY가 아니므로 컨테이너를 백그라운드로 띄우고 로그와 API로
확인한다. 웹 페이지는 작업자가 현장 PC 브라우저로 연다.

```bash
cd ~/fr5_infer
docker compose run -d --name fr5_infer infer --mode fake --backend replay   # 하드웨어 없음
docker compose run -d --name fr5_infer infer --mode shadow                  # 로봇 연결, 정지
docker compose run -d --name fr5_infer infer --mode real --confirm-hardware # 로봇 동작 (규칙 1)
docker logs fr5_infer 2>&1 | tail -50
docker stop fr5_infer && docker rm fr5_infer                                # 정상 종료
```

API는 다음과 같다(작업자가 웹에서 누르는 버튼과 같다).

```bash
curl -s 127.0.0.1:8080/api/status | python3 -m json.tool   # 상태 전체
curl -s -XPOST 127.0.0.1:8080/api/home                      # 홈 복귀 (real만)
curl -s -XPOST 127.0.0.1:8080/api/start                     # 정책 시작 (응답에 run_dir)
curl -s -XPOST 127.0.0.1:8080/api/stop                      # 정지
curl -s -XPOST 127.0.0.1:8080/api/reset                     # fault 리셋
```

- 한 번에 하나의 컨테이너만 띄운다. `docker ps -a`로 남은 `fr5_infer`·`fr5`가 없는지
  확인한다.
- 모델 로드에 1분 정도 걸린다. `/api/status`의 `backend`에 `loading`이 사라지고
  `gpu_memory_gb`가 보이면 준비가 끝난 것이다.
- config를 바꾼 뒤에는 컨테이너를 내렸다 다시 띄워야 반영된다.
- 상태 확인에 유용한 필드는 다음과 같다.
  - `robot.state`, `robot.reason`
  - `policy.state`, `policy.inferences`, `policy.last_latency_s`, `policy.latency_ema_s`
  - `policy.last_end_reason`
  - `cameras.<name>.age_ms`

## 3. 검증 절차

각 항목의 결과(통과/실패/보류)와 근거(명령 출력, 로그 줄, 작업자 답변)를 기록한다.

### 3-1. 준비 (로봇 동작 없음)

1. 사전 확인:

   ```bash
   id -u                     # 1000
   docker ps                 # 권한 오류 없이 동작, fr5 컨테이너 없음
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
   ```

   기대: `NVIDIA GeForce RTX 4090, 24564 MiB` 근처, 드라이버 570 이상. 실제 값을 기록한다.
2. nvidia-container-toolkit이 있는지 본다.

   ```bash
   docker run --rm --gpus all ubuntu nvidia-smi
   ```

   GPU 표가 나오면 3번으로 간다. `could not select device driver "nvidia"`가 나오면
   `README.md`의 "현장 PC 준비" 명령으로 설치한다. sudo 비밀번호는 작업자가 입력한다.
   설치 후 위 명령을 다시 실행한다.
3. 파일 배치(0장 표)와 이미지 로드:

   ```bash
   docker load < /path/to/fr5_infer_0.1.0.tar.gz
   docker image ls ghcr.io/benlee-robotics/fr5_infer        # 0.1.0
   cd ~/fr5_infer
   mkdir -p runs checkpoints docs/field_reports
   echo "INFER_TAG=0.1.0" > .env
   ls checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model/   # config.json, model.safetensors, tokenizer/ ...
   ls ~/fr5_teleop/datasets/fr5_pick_cube_20261027/                    # data meta videos fr5_conversion.json
   ```

   `runs/`를 먼저 만들지 않으면 Docker가 root 소유로 만들어 기록을 쓰지 못한다.
4. `~/fr5_teleop/config.yaml`(현장 config)을 확인한다. **고치지 않고 읽기만 한다.**
   - `robot.ip: 192.168.57.3`
   - `gripper.enabled: true` (real 모드는 이 값이 없으면 시작을 거부한다)
   - `recording.cameras`에 `wrist`, `top`과 각 serial
   - `home.tcp`
5. 이번 검증용으로 `~/fr5_infer/config.yaml`을 바꾼다. 작업자에게 알리고 보고서에 적는다.
   - `safety.max_velocity_mm_s: 50.0` (첫 실기는 저속, 3-7에서 단계적으로 올림)
   - `log.save_images: true` (개발 PC 분석용으로 추론 입력 이미지를 저장)

### 3-2. 이미지·GPU·설정 확인 (로봇 동작 없음)

```bash
cd ~/fr5_infer
docker compose run --rm -T --entrypoint python infer -c \
  "import torch, lerobot, transformers; print(torch.cuda.get_device_name(0), torch.__version__, lerobot.__version__, transformers.__version__)"
docker compose run --rm -T --entrypoint python infer -c \
  "from pathlib import Path; from fr5_infer.config import InferConfig; c=InferConfig.from_yaml(Path('config.yaml'), overrides={'mode':'real'}); t=c.teleop; print(t.robot_ip, t.gripper.enabled, [(k.name, k.serial) for k in c.cameras]); print(c.policy.checkpoint, c.max_velocity_mm_s, c.max_angular_velocity_deg_s, c.log.save_images)"
```

기대:

- 첫 명령: `NVIDIA GeForce RTX 4090 2.7.1+cu128 0.6.2 5.5.4`
- 둘째 명령 첫 줄: `192.168.57.3 True [('wrist', '<serial>'), ('top', '<serial>')]`
- 둘째 명령 둘째 줄: 체크포인트 경로, `50.0 15.0 True`

오류가 나면 메시지 원문을 기록한다. 그리퍼 비활성, 카메라 누락, 경로 오류는 여기서 걸린다.

### 3-3. 오프라인 평가 (로봇·카메라 없음, GPU)

```bash
docker compose run --rm -T --entrypoint python infer scripts/eval_offline.py \
  --checkpoint checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model \
  --dataset ../fr5_teleop/datasets/fr5_pick_cube_20261027 --episodes 0 40 80 \
  2>&1 | grep -E "^checkpoint|^loaded|All keys|^ep |^summary|^  [a-z]"
```

- 기대: `All keys loaded successfully!`, 마지막에 `summary` 표.
- `summary`를 0장의 개발 PC 값과 나란히 보고서에 적는다.
- 정책은 샘플링이 무작위라 값이 조금 다를 수 있다. 위치 오차가 ±5 mm, 그리퍼 일치율이
  ±0.03 안이면 같은 것으로 본다. 그보다 크게 나쁘면 3-4로 넘어가지 말고 작업자에게 알린다.
- **4090의 추론 지연**(`latency_ms_p50`, `latency_ms_max`)을 기록한다. 300 ms를 넘으면 보류로
  표시한다.

### 3-4. fake 모드 (로봇·카메라 없음)

1. 녹화 재생:
   1. `docker compose run -d --name fr5_infer infer --mode fake --backend replay --episode 1`
   2. 작업자가 브라우저에서 `http://127.0.0.1:8080/`을 연다. 상태 `IDLE (ready)`, 카메라
      2개에 합성 화면(움직이는 흰 막대)이 보이는지 확인한다.
   3. **정책 시작**을 누른다. 약 13초 동안 `RUNNING`이 되고, 중간에 `GRIPPER` →
      `STARTING` → `RUNNING`이 한 번 나타나며, 끝에 `IDLE (trajectory_exhausted)`가
      되는지 확인한다.
   4. `ls runs/`에 새 폴더가 있고 `meta.json`, `inferences.npz`, `ticks.npy`가 있는지 본다.
   5. `docker stop fr5_infer && docker rm fr5_infer`
2. 정책 로드:
   1. `docker compose run -d --name fr5_infer infer --mode fake`
   2. `/api/status`의 `backend.gpu_memory_gb`와 `load_s`를 기록한다.
   3. **정책 시작**을 누르고 10초 뒤 **정지**를 누른다. 카메라가 합성 화면이라 동작은
      의미 없다. `policy.latency_ema_s`, `inferences`, 에러 유무를 기록한다.
   4. `docker stop fr5_infer && docker rm fr5_infer`

### 3-5. shadow 모드 (로봇 연결, 로봇 동작 없음, 카메라 실제)

shadow는 홈 복귀를 하지 않는다. **먼저 로봇을 홈에 둔다.** 작업자에게 fr5_teleop의 홈
복귀(펜던트 또는 `/monitor` 홈으로)로 옮겨 달라고 하고, fr5_teleop 컨테이너를 내린다.

```bash
docker ps                                   # fr5 없음 확인
docker compose run -d --name fr5_infer infer --mode shadow
docker logs fr5_infer 2>&1 | grep -E "robot ready|camera .* streaming|pi05 warmup|ERROR"
```

| # | 항목 | 방법 | 기대 / 기록할 것 |
|---|---|---|---|
| S1 | 연결 | 로그와 `/api/status` | `robot ready mode=shadow tcp=...`가 홈 TCP와 1 mm, 1° 안. `robot.state` `IDLE`, reason `shadow_ready`. 카메라 두 대 `age_ms` < 100, `error` 없음. 카메라 serial은 로그 `camera <name> streaming:`에서 기록 |
| S2 | 관측 벡터 | 아래 명령 | 가이드 홈 벡터 `[0.393, -0.123, 0.504, 0.998, -0.057, -0.018, -0.057, -0.998, -0.010, ~0]`과 소수 둘째 자리까지 같은지 |
| S3 | 카메라 화면 | 작업자가 웹 미리보기 확인 | `top`이 작업대 전체(큐브, 박스), `wrist`가 그리퍼 앞을 보는지. 시연 녹화 때와 같은 위치·각도인지 작업자에게 묻는다(카메라가 옮겨졌으면 정책이 크게 틀린다) |
| S4 | 추론 | 큐브를 시연 때처럼 놓고 **정책 시작**, 15초 뒤 **정지** | `robot.state`가 `RUNNING`이 되는지, 아니면 `IDLE (start_offset_XXmm_YYdeg)`로 거부되는지. 거부되면 XX, YY 값 기록. `latency_ema_s`, `inferences` |
| S5 | 반복 | 큐브 위치를 2곳 더 바꿔 S4 반복 | 회당 결과 |
| S6 | 보냈을 명령 | 아래 분석 명령 | 정책이 처음 가려는 방향이 큐브 쪽인지 작업자와 함께 판단 |

S2 명령:

```bash
docker exec fr5_infer python -c "import json, urllib.request; from fr5_infer.conversion import encode_state; s=json.load(urllib.request.urlopen('http://127.0.0.1:8080/api/status')); print([round(float(v),3) for v in encode_state(tuple(s['robot']['tcp']), s['robot']['gripper']*100, 0, 100)])"
```

S6 명령(가장 최근 run을 분석한다. shadow에서는 로봇이 움직이지 않으므로 "보냈을 명령"이
홈에서 어디로 향하는지만 본다):

```bash
docker compose run --rm -T --entrypoint python infer -c "
import glob, json, numpy as np
r = sorted(glob.glob('runs/*/'))[-1]; m = json.load(open(r + 'meta.json'))
print(r, m.get('end_reason'), m.get('inference'))
t = np.load(r + 'ticks.npy'); c = t[t['cmd_valid'] == 1]
if len(c):
    s = (c['t_ns'] - c['t_ns'][0]) / 1e9; d = c['cmd'][:, :3] - c['tcp'][0, :3]
    for x in (0.5, 1, 2, 3, 5):
        i = np.searchsorted(s, x)
        if i < len(c): print(f'{x:>4}s  cmd - start (x, y, z mm):', np.round(d[i], 1))
i = np.load(r + 'inferences.npz'); print('gripper raw (step 0):', np.round(i['chunk'][:12, 0, 9], 2))
"
```

출력은 로봇 base 좌표의 변화량(mm)이다. 실제로 어느 방향인지는 큐브 위치와 비교해
작업자에게 확인받는다(예: 작업자가 큐브 쪽으로 TCP를 펜던트로 조금 옮겨 보고 x, y 부호를
대조한다).

종료: `docker stop fr5_infer && docker rm fr5_infer`

### 3-6. 실제 로봇: 홈 복귀와 녹화 재생 (규칙 1 적용)

`config.yaml`의 `safety.max_velocity_mm_s: 50.0`인지 확인하고, 다음 명령으로 실행한다.

```bash
docker compose run -d --name fr5_infer infer --mode real --confirm-hardware --backend replay --episode 0
```

| # | 항목 | 방법 | 기대 / 기록할 것 |
|---|---|---|---|
| R1 | 홈 복귀 | 로봇을 홈에서 10~20 cm 옮겨 둔 뒤 웹 **홈으로** | 직선으로 부드럽게 이동 후 정지. `robot.reason` `home_reached`. fr5_teleop 홈 복귀와 같은 느낌인지 작업자 체감 |
| R2 | 녹화 재생 | 작업대를 비우고(큐브 없이) **정책 시작** | 에피소드 0의 시연 궤적을 저속으로 따라감. 그리퍼가 닫혔다 열림(빈손). 끝에 `trajectory_exhausted`로 멈춤. 동작이 부드러운지, 그리퍼 전후 멈춤이 짧은지 작업자 체감 |
| R3 | 정지 | R2를 다시 시작하고 5초쯤 뒤 웹 **정지** (또는 Esc) | 즉시 멈춤, `IDLE (stopped_by_operator)`. 정지 순간 튐 여부 |
| R4 | 시작 거부 | 로봇을 홈에서 5 cm 이상 옮겨 둔 채 **정책 시작** | 움직이지 않고 `start_offset_...`로 거부됨 |

R2의 run 폴더로 S6 분석 명령을 실행해 결과를 붙인다. 이번에는 로봇이 실제로 움직이므로,
`cmd`와 실제 `tcp`의 차이도 다음 명령으로 본다.

```bash
docker compose run --rm -T --entrypoint python infer -c "
import glob, numpy as np
r = sorted(glob.glob('runs/*/'))[-1]; t = np.load(r + 'ticks.npy'); c = t[t['cmd_valid'] == 1]
e = np.linalg.norm(c['cmd'][:, :3] - c['tcp'][:, :3], axis=1); dt = np.diff(t['t_ns']) / 1e6
print(r, 'cmd-tcp mm p50 %.1f p99 %.1f max %.1f' % (*np.percentile(e, [50, 99]), e.max()), '| tick ms p99 %.2f max %.2f' % (np.percentile(dt, 99), dt.max()))
"
```

종료: `docker stop fr5_infer && docker rm fr5_infer`

### 3-7. 실제 로봇: 정책 실행 (규칙 1, 2 적용)

시작 전 작업자에게 규칙 2의 내용을 알리고 동의를 받는다. 다음 명령으로 실행한다.

```bash
docker compose run -d --name fr5_infer infer --mode real --confirm-hardware
```

1. 웹 **홈으로**로 홈에 간다. 큐브를 시연 때처럼 놓고, 박스도 시연 위치에 둔다.
2. **정책 시작.** 다음 둘 중 하나가 된다.
   - `RUNNING`이 되면 3번으로 간다.
   - `start_offset_XXmm_YYdeg`로 거부되면 그 값을 기록한다. 같은 조건으로 2회 더 시도한다
     (매번 값이 다르다). 3회 모두 거부되고 XX가 40 이하, YY가 10 이하면, 작업자 동의를 받아
     `safety.max_start_offset_mm: 40.0`, `max_start_offset_deg: 10.0`으로 바꾸고 다시
     띄운다. 바꾼 값을 보고서에 적는다. 그보다 크면 실기 실행을 보류하고 5장으로 간다.
3. 실행을 관찰한다. 작업자는 정지 버튼을 누를 준비를 하고 있는다.
   - 큐브 쪽으로 가는가
   - 그리퍼를 닫는가
   - 박스 쪽으로 옮기는가
   - 놓는가
4. 끝나면(`trajectory_exhausted`, `max_episode_time`, 작업자 정지) 결과를 기록한다.
   - 결과: 성공 / 부분(어디까지) / 실패
   - 끝난 이유(`policy.last_end_reason`)
   - run 폴더
   - 작업자 관찰
5. 큐브 위치를 바꿔 총 3회 실행한다. 회마다 홈으로 돌아간 뒤 시작한다.
6. 50 mm/s에서 이상 동작이 없었고 작업자가 원하면 `safety.max_velocity_mm_s: 100.0`으로
   올려 2회 더 한다. 바꾼 값을 보고서에 적는다.

각 회의 run 폴더로 3-6의 `cmd-tcp` 분석 명령도 실행해 붙인다. 관찰할 점은 다음과 같다.

- 정책이 그리퍼를 너무 일찍 또는 늦게 닫는지
- 같은 자리에서 머뭇거리는지(앞뒤로 흔들림)
- 시연보다 훨씬 느려서 정책이 헷갈리는 것처럼 보이는지

## 4. 문제 해결

| 증상 | 확인 순서 |
|---|---|
| `could not select device driver "nvidia"` | nvidia-container-toolkit 미설치 또는 `nvidia-ctk runtime configure` 후 docker 재시작 안 함 (3-1의 2) |
| `CUDA driver version is insufficient` | 드라이버가 570 미만. `nvidia-smi` 드라이버 버전 기록 후 작업자에게 알린다 |
| `gripper.enabled must be true in the teleop config` | 3-2에서 real 설정을 읽을 때 남. 현장 `~/fr5_teleop/config.yaml`의 `gripper.enabled`가 `false`. 작업자에게 확인 후 fr5_teleop 현장 값으로 맞춘다 |
| `policy.checkpoint ... not found`, `config.json` 없음 | 체크포인트 경로가 `checkpoints/fr5_pick_cube_pi05_expert/002500/pretrained_model`인지 |
| `policy.conversion not found` | 데이터셋이 `~/fr5_teleop/datasets/fr5_pick_cube_20261027/`에 있는지(compose가 `../fr5_teleop/datasets`를 마운트) |
| `cameras [...] are not in the teleop config` | 현장 `~/fr5_teleop/config.yaml`의 `recording.cameras` 이름이 `top`, `wrist`인지 |
| 카메라 `error` / `age_ms` 큼 | serial 오타, USB 3(`lsusb -t`의 `5000M`), fr5_teleop 컨테이너가 카메라를 잡고 있지 않은지 |
| `connect_failed` (FAULT) | fr5_teleop 컨테이너가 떠 있는지(`docker ps`), `ping 192.168.57.3`, 컨트롤러 모드 |
| `start_refused_*` | `start_refused_running` 등: 상태가 `IDLE`이 아님. `start_refused_outside_workspace`: TCP가 workspace 밖 |
| `start_timeout` | 15초 안에 첫 궤적이 안 옴. `policy.state`가 `waiting_for_camera_*`이면 카메라, `waiting_for_robot_state`이면 로봇 상태 문제 |
| `trajectory_exhausted`가 실행 중간에 자주 남 | 추론 지연이 청크보다 김. `latency_ema_s`, `dropped_points` 기록 |
| `gripper_timeout` / `gripper_fault` (FAULT) | 그리퍼 전원·통신. 웹 **fault 리셋** 후 홈으로. 로그 원문 기록 |
| `runs/`에 쓰기 실패 (`Permission denied`) | `ls -ld runs`가 root 소유면 작업자에게 `sudo chown 1000:1000 runs` 요청 |
| 웹 페이지가 안 열림 | `docker logs fr5_infer`에 `operator page:`가 나왔는지(모델 로드 후에 열림), 다른 프로그램이 8080을 쓰는지 |

## 5. 보고서와 전달

1. `~/fr5_infer/docs/field_reports/<YYYY-MM-DD>_pi05_002500.md`에 작성한다.
   - 환경: OS, 커널, Docker 버전, GPU·드라이버, nvidia-container-toolkit 버전
     (`nvidia-ctk --version`), 이미지 ID
     (`docker image inspect ghcr.io/benlee-robotics/fr5_infer:0.1.0 -f '{{.Id}}'`)
   - 3-1 ~ 3-7 항목별 표: 항목, 결과, 근거
   - 3-3 오프라인 평가 `summary`와 개발 PC 값 비교, 4090 추론 지연
   - 3-5, 3-7의 시작 오프셋 값 전체(거부·수락 모두)
   - 3-7 회별 결과와 작업자 관찰
   - 바꾼 config 값(`~/fr5_infer/config.yaml`, 3-1의 5와 3-7)
   - 실패/보류 항목: 증상, 재현 방법, 로그 원문
   - 개발 PC 후속 작업 제안(예: 다음 체크포인트, 시작 오프셋, 속도 설정)
2. 작업자에게 요약을 보여 준다.
3. `runs/`를 묶어 보고서와 함께 전달한다. `save_images`로 추론 입력 이미지가 들어 있어
   개발 PC에서 다시 분석할 수 있다.

   ```bash
   tar czf ~/fr5_infer_runs_<날짜>.tgz -C ~/fr5_infer runs docs/field_reports
   ```
