# 현장 검증 작업 지시서 (fr5_infer 0.1.2, Pi0.5 체크포인트 005000)

이 문서는 현장 PC에서 작업하는 Claude에게 주는 지시서다. 작업자(사람)와 함께
아래 순서대로 검증하고, 결과를 보고서로 남긴다. 작업자와는 한국어로 대화한다.

## 0. 배경

- **프로젝트:** fr5_infer는 텔레옵 시연으로 학습한 LeRobot Pi0.5 정책을 FR5 실기에서
  실행한다. 조작은 웹 페이지(`http://127.0.0.1:8080/`)에서 한다. 구조와 설정은
  `README.md`를 본다.
- **fr5_teleop과의 관계:** 로봇 클라이언트·안전 제한·관측 변환은 fr5_teleop **v0.4.3** 코드를
  그대로 쓴다(이미지에 포함). 로봇 IP, 그리퍼, 홈 자세, workspace, 카메라 serial은
  **`~/fr_ws/fr5_teleop/config.yaml`(현장 config)** 에서 읽는다. fr5_infer는 그 config와 데이터셋만
  마운트한다. 현장 fr5_teleop 저장소(v0.4.2)도 3-1에서 v0.4.3으로 올린다. 0.4.2로 텔레옵을 띄우면
  텔레옵에서도 그리퍼 동작마다 `gripper_timeout`이 나기 때문이다.
- **배포 방식:**
  - 저장소: https://github.com/benlee-robotics/fr5_infer, 태그 `v0.1.2`(`main`).
  - Docker 이미지: `ghcr.io/benlee-robotics/fr5_infer:0.1.2`
    (digest는 3-1에서 기록, public)
  - 현장 스크립트(`scripts/gripper_check.py`, `scripts/analyze_run.py`)는 저장소 것을 쓰도록
    실행할 때 `scripts/`를 마운트한다(2장).
- **현장 PC:** 2026-10-29 검증과 같다. RTX 4090, 드라이버 595.99.02, nvidia-container-toolkit
  설치됨. 저장소 위치는 `~/fr_ws/fr5_infer`, `~/fr_ws/fr5_teleop`.
- **로봇:** FR5, IP `192.168.57.3`, 그리퍼 사용, 카메라 `wrist`·`top`(D435).
- **정책:** `fr5_pick_cube_pi05_expert` **005000** (10-29와 같음). 현장 PC에 이미 있다
  (`checkpoints/fr5_pick_cube_pi05_expert/005000/pretrained_model`).
  - 데이터셋 `fr5_pick_cube_20261027`, 지시문 `pick up the cube and place it in the box`
  - 30 Hz, 50스텝(1.67 s) 청크

### 지난 두 번의 현장과 이번에 바뀐 것

| 버전 | 그리퍼 완료 판정 | 결과 |
|---|---|---|
| v0.1.0 (teleop 0.4.1) | 결과를 무시하고 항상 완료 | 명령 약 20 ms 뒤 "완료". 그 뒤 서보 재시작(`ServoMoveStart`)이 실제 동작이 끝날 때까지 1.1~1.6 s 막혔다. 10-29 회 1·회 3의 `trajectory_exhausted` 원인 |
| v0.1.1 (teleop 0.4.2) | SDK 문서대로 `status == 1`만 완료 | 이 그리퍼는 `[1, status]`를 돌려주고, status가 0 움직임 / 2 물체 잡고 멈춤 / 3 목표 도착이다. 모든 동작이 5 s 뒤 `gripper_timeout`. 3-2에서 발견해 실기 전에 멈췄다 |
| **v0.1.2 (teleop 0.4.3)** | **status가 0이 아니면 완료**, fault는 CNDE `gripper_fault` | 현장 값(`[1, 0]` → `[1, 3]` 또는 `[1, 2]`)으로 개발 PC에서 재현 시험함. 실기 확인은 이번이 처음 |

v0.1.2에서 함께 바뀐 것:

| 항목 | 내용 |
|---|---|
| **완료 뒤 위치 안정 대기** (`gripper.settle_s` 0.15, `max_settle_s` 0.5) | CNDE `gripper_position`은 움직이는 동안 0이고, 완료 0.1~0.3 s 뒤에 최종값이 된다. 완료 뒤 측정 위치가 0.15 s 동안 그대로일 때(최대 0.5 s) 서보를 다시 시작한다. 그래서 정책은 그리퍼 직후 관측에서 최종 위치를 본다 |
| `gripper_check.py` | 판정 옆에 `GetGripperMotionDone` 원래 값과 CNDE 그리퍼 필드, 위치가 멈춘 시각을 함께 출력한다 |

v0.1.1에서 추가된 것(그대로, 기본 꺼짐):

| 항목 | 내용 |
|---|---|
| **RTC** (`policy.rtc.enabled`, 기본 꺼짐) | lerobot의 추론 시점 guided Real-Time Chunking. 새 청크가 이전 청크 중 아직 실행하지 않은 부분을 이어 가도록 유도한다. 재학습이 필요 없다. 지연이 약 30 ms 늘어난다 |
| **시간 앙상블** (`execution.ensemble`, 기본 꺼짐) | ACT 방식. 시각마다 그 시각을 예측한 청크들의 가중 평균을 낸다. 그리퍼 출력도 평균한다 |
| 기록 | `inferences.npz`에 `rtc_prefix_steps`, `rtc_delay_steps` 추가 |

### 이번 검증의 목적

1. **그리퍼 수정 확인:** 빈손(status 3)과 큐브를 잡았을 때(status 2) 모두 완료가 실제 동작
   끝에 맞춰 보고되는지, fault가 0인지, 완료 뒤 위치가 최종값으로 안정되는지.
2. **기준선(RTC·앙상블 끔):** 그리퍼 수정만으로 10-29의 `trajectory_exhausted`, 그리퍼 뒤 멈춤이
   사라졌는지, 100 mm/s jerking이 남는지.
3. **RTC:** 같은 조건에서 RTC만 켰을 때 jerking과 청크 이음매가 줄어드는지, 성공률이 유지되는지.
4. **(시간이 남으면) 앙상블:** 오프라인에서는 RTC보다 못했다. 확인용이다.

### 비교 기준

2026-10-29 현장(005000, v0.1.0). `docs/field_reports/2026-10-29_pi05_005000.md`에 자세히 있다.

| 항목 | 값 |
|---|---|
| 정책 성공 | 50 mm/s 0/2, 75 mm/s 3/3, 100 mm/s 1/1 |
| 그리퍼 뒤 첫 RUNNING tick | 1.1 s (모든 그리퍼 동작), 닫자마자 열 때 1.6 s → `trajectory_exhausted` |
| 그리퍼 직후 관측 gripper | 항상 0.00 |
| 청크 이음매 튐 (`analyze_run.py`) | 회 1 (50 mm/s) p50 29.5 / max 69.2 mm, 회 3 (100 mm/s) p50 19.8 / max 66.4 mm |
| cmd−tcp 최대 | 75 mm/s 7.3 mm, 100 mm/s 11.6 mm |
| 추론 지연 p50 | 120~136 ms |
| 작업자 체감 | 75 mm/s가 가장 부드러움, 100 mm/s "엄청 jerking" |

2026-10-29 v0.1.1 점검(`docs/field_reports/2026-10-29_pi05_005000_v0.1.1.md`):

| 항목 | 값 |
|---|---|
| 그리퍼 SDK 직접 진단 | 빈손 열기·닫기 `[1, 0]` → `[1, 3]` 1.78·1.82 s, 큐브 잡기 `[1, 2]` 1.28 s(위치 47 → 59), 놓기 1.32 s. CNDE `gripper_fault` 항상 0 |
| shadow RTC (4090) | 지연 p50 135.5 → 169.4 ms, 이음매 튐 p50 49.3 → 9.5 mm, guided 15/16 |

개발 PC 오프라인(002500, 녹화 에피소드 0/40/80 순차 재생, RTX 5080, `scripts/eval_chunk_continuity.py`):

| 방식 | 이음매 튐 p50 / max | 위치 오차 | 지연 p50 |
|---|---|---|---|
| 독립 샘플 (끔) | 35.2 / 155.8 mm | 28.4 mm | 126 ms |
| 앙상블 | 11.1 / 40.7 mm | 32.2 mm | 126 ms |
| RTC (horizon 10) | 3.4 / 12.9 mm | 26.6 mm | 155 ms |
| RTC + 앙상블 | 6.3 / 31.0 mm | 32.0 mm | 155 ms |

### 이미 알려진 사항 (문제로 보고하지 않는다)

- 모델 로드 때 `Vision embedding key might need handling: ...patch_embedding...` 경고 2줄.
  `All keys loaded successfully!`가 나오면 정상이다.
- 그리퍼가 움직일 때 서보가 멈췄다 다시 시작하는 것은 설계다(`GRIPPER` → `STARTING` →
  `RUNNING`). **v0.1.2에서는 `GRIPPER` 상태가 약 1.4~2.3 s 이어진다**(동작 1.3~1.8 s +
  위치 안정 대기 0.15~0.5 s). 10-29 v0.1.0에서는 20~30 ms였고, 이번에 길어진 것이 정상이다.
- 놓은 뒤 정책이 멈추지 않고 홈 근처를 배회한다(종료 판단 없음). 이번 검증 대상이 아니다.
- 시작 오프셋 거부(`start_offset_XXmm_YYdeg`)가 가끔 난다. 한도는 10-29처럼 40 mm / 10°로 둔다.
- 궤적이 끝나면 서보를 멈춘다(`trajectory_exhausted`). 실행 **끝**에 나는 것은 설계다. 그리퍼
  직후에 나는 것은 이번에 고친 문제이므로 기록한다.

## 1. 반드시 지킬 규칙

1. **로봇이나 그리퍼가 움직일 수 있는 실행(`--mode real`, `gripper_check.py`)은 매번
   작업자에게 먼저 알리고 동의를 받은 뒤 실행한다.** 작업자가 비상정지 버튼 옆에 있는지
   확인한다. 정책 실행 때는 웹의 **정지** 버튼(또는 Esc)을 작업자가 누를 수 있게 페이지를
   열어 둔다.
2. **그리퍼 판정 방식이 바뀐 뒤 처음 하는 실기다.** 3-2(그리퍼 단독 점검)를 통과하기 전에는
   정책을 실기로 실행하지 않는다.
3. **로봇에 연결하는 프로그램은 한 번에 하나만 띄운다.** fr5_teleop 컨테이너(`fr5`), 앱
   컨테이너(`fr5_infer`), `gripper_check.py`를 동시에 띄우지 않는다. 실행 전마다 `docker ps`로
   확인한다. `analyze_run.py`는 로봇에 연결하지 않으므로 앱이 떠 있어도 된다.
4. Claude는 로봇과 화면을 직접 볼 수 없다. 움직임, 부드러움, 그리퍼 동작, 웹 화면 표시는
   작업자에게 구체적으로 물어보고 답을 그대로 기록한다. 추측으로 "통과"라고 적지 않는다.
5. **`fr5_infer/` 코드, `scripts/`, 이미지는 수정하지 않는다.** 문제를 찾으면 증상, 로그, 재현
   방법을 보고서에 적는다. `config.yaml` 두 개는 이 문서가 지시하거나 작업자가 요청할 때만
   고치고, 바꾼 값과 이유를 보고서에 적는다.
6. 이상 동작(급가속, 반대 방향, 크게 도는 회전, 멈추지 않음, 테이블·박스로 파고듦)이 보이면
   다음 순서로 대응한다.
   1. 즉시 작업자에게 정지를 요청한다(웹 정지 / Esc).
   2. Claude는 `curl -s -XPOST 127.0.0.1:8080/api/stop`을 보낸다.
   3. 그래도 멈추지 않으면 `docker stop fr5_infer`.

   비상정지는 작업자가 판단한다.
7. 커밋과 push는 작업자가 요청할 때만 한다. 체크포인트, 데이터셋, `runs/`는 git에 넣지
   않는다. 현장 PC에서 이미지를 빌드하거나 push하지 않는다.

## 2. 실행 방식 (Claude용)

Claude의 셸은 대화형 TTY가 아니므로 컨테이너를 백그라운드로 띄우고 로그와 API로 확인한다.
웹 페이지는 작업자가 현장 PC 브라우저로 연다. 모든 명령은 `~/fr_ws/fr5_infer`에서 실행한다.

```bash
cd ~/fr_ws/fr5_infer
docker compose run -d --name fr5_infer infer --mode shadow                  # 로봇 연결, 정지
docker compose run -d --name fr5_infer infer --mode real --confirm-hardware # 로봇 동작 (규칙 1)
docker logs fr5_infer 2>&1 | tail -50
docker stop fr5_infer && docker rm fr5_infer                                # 정상 종료
```

현장 스크립트는 `scripts/`를 마운트해서 실행한다. 셸 변수는 명령 사이에 유지되지 않으므로
`-v` 옵션을 매번 그대로 쓴다.

```bash
docker compose run --rm -T -v "$PWD/scripts:/ws/fr5_infer/scripts:ro" --entrypoint python infer scripts/gripper_check.py close   # 그리퍼 (규칙 1, 3)
docker compose run --rm -T -v "$PWD/scripts:/ws/fr5_infer/scripts:ro" --entrypoint python infer scripts/analyze_run.py runs/<폴더>
```

API(작업자가 웹에서 누르는 버튼과 같다):

```bash
curl -s 127.0.0.1:8080/api/status | python3 -m json.tool   # 상태 전체
curl -s -XPOST 127.0.0.1:8080/api/home                      # 홈 복귀 (real만)
curl -s -XPOST 127.0.0.1:8080/api/start                     # 정책 시작 (응답에 run_dir)
curl -s -XPOST 127.0.0.1:8080/api/stop                      # 정지
curl -s -XPOST 127.0.0.1:8080/api/reset                     # fault 리셋
```

- 모델 로드에 1분 정도 걸린다. `/api/status`의 `backend`에 `gpu_memory_gb`가 보이면 준비가
  끝난 것이다. RTC를 켜면 `backend.rtc`에 설정이, `backend.rtc_warmup_s`에 값이 보인다.
- **config를 바꾼 뒤에는 컨테이너를 내렸다 다시 띄워야 반영된다.** RTC는 모델을 로드할 때
  설정된다. 조건(3-4 ~ 3-6)을 바꿀 때마다 재시작한다.

## 3. 검증 절차

각 항목의 결과(통과/실패/보류)와 근거(명령 출력, 로그 줄, 작업자 답변)를 기록한다.

### 3-1. 갱신 (로봇 동작 없음)

1. 10-29에 바꾼 현장 config를 보관하고 저장소를 갱신한다.

   ```bash
   cd ~/fr_ws/fr5_infer
   docker ps -a                                   # fr5, fr5_infer 없음
   git status --short                             # 바뀐 파일 기록
   cp config.yaml ~/config_infer_before_v0.1.2.yaml
   git diff config.yaml > ~/config_infer_before_v0.1.2.diff
   git checkout -- config.yaml
   # v0.1.1 점검 보고서는 이제 저장소에 들어 있다. 미추적 파일이 checkout을 막으므로 옮겨 둔다.
   mv docs/field_reports/2026-10-29_pi05_005000_v0.1.1.md ~/2026-10-29_pi05_005000_v0.1.1.local.md
   git fetch --tags origin && git checkout main && git pull --ff-only
   git describe --tags                            # v0.1.2
   echo "INFER_TAG=0.1.2" > .env
   docker compose pull
   docker image inspect ghcr.io/benlee-robotics/fr5_infer:0.1.2 -f '{{index .RepoDigests 0}}'
   ```

   digest를 기록한다. `diff ~/2026-10-29_pi05_005000_v0.1.1.local.md
   docs/field_reports/2026-10-29_pi05_005000_v0.1.1.md`가 비어 있는지 확인한다. 차이가 있으면
   작업자에게 알리고 로컬 파일을 보관한다.
2. 현장 fr5_teleop 저장소를 v0.4.3으로 올린다. 현장 `config.yaml` 수정은 유지한다
   (0.4.2 → 0.4.3에서 config 키는 바뀌지 않았다).

   ```bash
   cd ~/fr_ws/fr5_teleop
   git status --short                             # config.yaml 외 변경이 있으면 작업자에게 확인
   cp config.yaml ~/config_teleop_field_v0.4.2.yaml
   git checkout -- config.yaml
   git fetch --tags origin && git checkout v0.4.3
   cp ~/config_teleop_field_v0.4.2.yaml config.yaml
   git diff --stat                                # config.yaml만 바뀜
   cat .env 2>/dev/null                           # 다른 값이 있으면 유지한다
   grep -q '^TELEOP_TAG=' .env 2>/dev/null && sed -i 's/^TELEOP_TAG=.*/TELEOP_TAG=0.4.3/' .env \
     || echo "TELEOP_TAG=0.4.3" >> .env
   cd ~/fr_ws/fr5_infer
   ```

   텔레옵 이미지 pull과 실행은 이번 검증에 필요 없다. 작업자가 나중에 텔레옵을 쓸 때
   `docker compose pull`한다.
3. 새 `config.yaml`에 현장 값을 넣는다. 보관한 diff와 비교해 작업자에게 확인받는다.
   - `policy.checkpoint: checkpoints/fr5_pick_cube_pi05_expert/005000/pretrained_model` (기본값, 확인만)
   - `safety.max_velocity_mm_s: 75.0`
   - `safety.max_start_offset_mm: 40.0`, `safety.max_start_offset_deg: 10.0`
   - `log.save_images: true`
   - `policy.rtc.enabled: false`, `execution.ensemble: false` (기본값 그대로, 확인만)
   - `gripper.settle_s: 0.15`, `gripper.max_settle_s: 0.5` (새 키, 기본값 그대로, 확인만)
4. 버전과 설정을 확인한다.

   ```bash
   docker compose run --rm -T --entrypoint python infer -c "
   import fr5_infer, teleop, inspect, teleop.robot.fairino_client as m
   from pathlib import Path; from fr5_infer.config import InferConfig
   c = InferConfig.from_yaml(Path('config.yaml'), overrides={'mode': 'real'})
   print(fr5_infer.__version__, teleop.__version__, '_gripper_fault' in inspect.getsource(m.FairinoRobotClient.get_gripper_motion_state))
   print(c.policy.checkpoint.parent.name, c.max_velocity_mm_s, c.safety.max_start_offset_mm, c.log.save_images, c.policy.rtc.enabled, c.execution.ensemble)
   g = c.teleop.gripper; print('gripper', g.enabled, g.open_position, g.closed_position, g.velocity, g.force, g.action_timeout_s, g.poll_period_s, c.gripper.settle_s, c.gripper.max_settle_s)"
   ```

   기대: `0.1.2 0.4.3 True`, `005000 75.0 40.0 True False False`, 그리퍼 줄 끝이 `0.15 0.5`.
   그리퍼 줄은 그대로 기록한다.

### 3-2. 그리퍼 단독 점검 (그리퍼만 움직임, 규칙 1·2·3)

팔은 움직이지 않는다. `gripper_check.py`가 로봇에 연결해 오류 리셋·enable·그리퍼 활성화를
하고(`--mode real` 시작과 같다), 그리퍼 명령 하나를 보낸 뒤 완료 판정을 20 ms마다 읽는다.
**작업자에게 그리퍼 사이에 손을 넣지 말라고 알린다.**

| # | 항목 | 방법 | 기대 |
|---|---|---|---|
| G1 | 빈손 닫기·열기 | `close`, `open`을 번갈아 3번씩 | `MoveGripper call returned` 100 ms 미만, `first poll ... done=False`, raw `[1, 0]` → `[1, 3]`, `done after` 약 1.8 s, `fault=0`, 닫힘 위치 약 89, 열림 약 0~1, `unchanged since`가 `done after`보다 0.3 s 이내 뒤 |
| G2 | 완료 시점 | 작업자에게 G1 중 1회를 보며 "그리퍼가 멈춘 순간"과 출력의 `done after`가 비슷한지 묻는다 | 완료가 그리퍼 멈춤보다 크게 빠르지 않음 |
| G3 | 큐브 쥐기 | 작업자가 **펜던트로** TCP를 내려 테이블 위 큐브가 손가락 사이에 오게 한다(손으로 들고 있지 않는다). `close` → `open`, 2회 | 닫기: raw `[1, 2]`, `done after` 약 1.3 s, 위치 약 59~60, `fault=0`. 열기: `[1, 3]`, 위치 약 0~1 |
| G4 | 원위치 | 작업자가 펜던트로 TCP를 큐브 위로 올린다. 홈 복귀는 3-4 시작 때 웹으로 한다 | – |

명령(한 번에 하나, 출력 전체를 보고서에 붙인다):

```bash
docker ps                                     # fr5, fr5_infer 없음
docker compose run --rm -T -v "$PWD/scripts:/ws/fr5_infer/scripts:ro" --entrypoint python infer scripts/gripper_check.py close
docker compose run --rm -T -v "$PWD/scripts:/ws/fr5_infer/scripts:ro" --entrypoint python infer scripts/gripper_check.py open
```

판정:

- **`NOT DONE`이 한 번이라도 나오면** 정책은 그 동작에서 `gripper_timeout`으로 FAULT가 난다.
  raw 값과 함께 기록하고 **3-4 이후 실기를 보류한 뒤** 5장으로 간다.
- **첫 poll이 `done=True`이거나 `done after`가 0.1 s 미만**: 이전 동작의 완료가 남아 있는 것이다.
  v0.1.0과 같은 문제가 남는다. 출력을 기록하고, 3-4 기준선은 진행하되 보고서에 표시한다.
- **`fault`가 0이 아니면** 즉시 멈추고 작업자에게 그리퍼 상태를 확인받는다. CNDE `gripper_fault`
  값의 뜻: 1 485 통신 시간 초과, 2 명령 오류, 3 물체 떨어짐, 그 밖 그리퍼 오류 코드.
- `unchanged since`가 `done after`보다 0.5 s 넘게 늦으면 기록한다(안정 대기 상한 0.5 s를
  넘는다는 뜻이다. 실기는 진행한다).

### 3-3. shadow RTC 지연 (이번에는 하지 않는다)

v0.1.1 점검에서 마쳤다(0장 비교 기준). 이번 변경은 그리퍼 쪽뿐이라 다시 하지 않는다.

### 3-4 ~ 3-6. 실기 정책 실행 (규칙 1, 2)

**공통 방법**

- 큐브 위치 3곳을 정해 테이프로 표시하고 모든 조건에서 같은 순서로 쓴다. 10-29에 성공한 곳
  위주로 작업자와 정한다.
  - P1: +X, +Y
  - P2: +X, −Y
  - P3: 작업자 선택
  - 박스 위치도 고정한다.
- 한 조건 안에서 75 mm/s로 P1→P2→P3을 하고, `safety.max_velocity_mm_s: 100.0`으로 바꿔
  다시 띄운 뒤 P1→P2→P3을 한다. 조건당 6회다.
- 매 회 순서:
  1. 웹 **홈으로**.
  2. 큐브를 놓고 작업자가 확인한다.
  3. **정책 시작**.
  4. 놓은 뒤 배회가 시작되면 약 5초 뒤 작업자가 **정지**한다.
  5. 결과를 기록한다.
- 이상 동작이 있으면 규칙 6을 따르고, 그 조건의 남은 회차는 작업자와 상의한다.
- 매 회 끝나면 `analyze_run.py runs/<그 회 폴더>`를 실행하고, 작업자에게 다음을 묻는다.
  - 결과: 성공 / 부분(어디까지) / 실패
  - **jerking 체감: 0 없음, 1 약간, 2 뚜렷, 3 심함**
  - 그 밖에 본 것

**기록 표** (회마다 한 줄)

| 조건 | 속도 | 위치 | run | 결과 | 끝난 이유 | jerking (0~3) | 그리퍼 뒤 첫 RUNNING tick (ms) | 그리퍼 직후 obs gripper | 이음매 튐 p50/max (mm) | cmd−tcp max (mm) | 지연 p50 (ms) | RTC guided |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

`analyze_run.py`의 `gripper actions` 줄에서 값을 옮긴다.
- 그리퍼 뒤 첫 RUNNING tick: 1.1 s(10-29)였던 값이 수십 ms로 줄었는지 본다.
- 그리퍼 직후 obs gripper: 큐브를 잡은 뒤 약 0.59, 연 뒤 약 0.0~0.01이면 정상이다.
  10-29 v0.1.0에서는 잡은 뒤에도 0.00이었다.
- `GRIPPER` 시간: 약 1.4~2.3 s가 정상이다.

#### 3-4. 조건 A: RTC·앙상블 끔 (기준선)

`policy.rtc.enabled: false`, `execution.ensemble: false`.

```bash
docker compose run -d --name fr5_infer infer --mode real --confirm-hardware
```

볼 것:
- 그리퍼 직후 `trajectory_exhausted`가 사라졌는지
- 그리퍼 뒤 첫 RUNNING tick
- 100 mm/s jerking이 10-29처럼 남는지

#### 3-5. 조건 B: RTC만 켬

`policy.rtc.enabled: true`, `execution.ensemble: false`. 3-4와 같은 순서로 한다.

볼 것:
- 이음매 튐과 jerking이 A보다 줄었는지(특히 100 mm/s)
- 성공률이 A와 비슷한지
- `RTC guided`가 그리퍼 직후와 첫 추론을 뺀 전부인지

#### 3-6. 조건 C·D (시간이 남을 때만)

- **C: 앙상블만** (`policy.rtc.enabled: false`, `execution.ensemble: true`). 75 mm/s P1~P3만 해도 된다.
  반응이 늦어 잡는 위치가 어긋나는지 본다.
- **D: 둘 다 켬.** C까지 하고도 시간이 남을 때만 한다.

끝나면 `config.yaml`을 조건 A 값(둘 다 `false`, 75 mm/s)으로 되돌린다.

## 4. 문제 해결

| 증상 | 확인 순서 |
|---|---|
| `config: unknown keys ...` 또는 `policy.rtc: unknown keys` | 이미지가 옛 버전이다. `.env`의 `INFER_TAG=0.1.2`, `docker compose pull` |
| `policy.rtc needs the pi05 backend` | `--backend replay`와 RTC를 같이 켰다. replay는 이번에 쓰지 않는다 |
| `gripper_timeout` (FAULT) | 3-2 G3에서 `NOT DONE`이 났는지. 웹 **fault 리셋** → 홈으로. `docker logs fr5_infer` 원문 기록 |
| `gripper_fault` (FAULT) | 그리퍼 전원·통신. 작업자 확인, 로그 원문 기록 |
| 그리퍼 직후 `trajectory_exhausted` | 0.4.1 문제가 남은 것이다. `analyze_run.py`의 `first RUNNING tick` 값과 run 폴더 기록 |
| RTC를 켜면 지연이 300 ms를 넘음 | `analyze_run.py`의 `latency`, `backend.rtc_warmup_s` 기록. 조건 B 보류 |
| `start_offset_XXmm_YYdeg` 반복 | 같은 조건으로 다시 시작(매번 샘플링이 다르다). 3회 연속이면 값과 함께 기록 |
| `connect_failed` (FAULT) | `docker ps`에 `fr5`나 다른 `fr5_infer`가 있는지, `ping 192.168.57.3` |
| 카메라 `error` / `age_ms` 큼 | USB 3 연결(`lsusb -t`의 `5000M`), 다른 컨테이너가 카메라를 잡고 있지 않은지 |
| `runs/` 쓰기 실패 | `ls -ld runs`가 root 소유면 작업자에게 `sudo chown 1000:1000 runs` 요청 |

## 5. 보고서와 전달

1. `~/fr_ws/fr5_infer/docs/field_reports/<YYYY-MM-DD>_pi05_005000_v0.1.2.md`에 작성한다.
   - 요약: 그리퍼 수정 판정, 조건별 성공률과 jerking 평균, 권장 조건
   - 환경: 이미지 digest, `git describe --tags`, 현장 fr5_teleop 저장소 버전
   - 바꾼 config 값(3-1의 3, 3-4 ~ 3-6)
   - 3-2 그리퍼 점검 출력 전체와 작업자 답변
   - 3-4 ~ 3-6 기록 표, 조건별로 묶은 요약(성공 n/6, jerking 평균, 튐 p50 평균, cmd−tcp max)
   - 실패·보류 항목: 증상, 재현 방법, 로그 원문
   - 개발 PC 후속 작업 제안
2. 작업자에게 요약을 보여 준다.
3. `runs/`와 보고서를 묶고, 작업자에게 NAS `rv_member/이병현/fr5_vla`에 올려 달라고 요청한다.

   ```bash
   tar czf ~/fr5_infer_runs_<날짜>.tgz -C ~/fr_ws/fr5_infer runs docs/field_reports
   ```
