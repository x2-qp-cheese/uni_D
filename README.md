# uni_D

Uni-D 배터리 열폭주 1~6단계 분류를 위한 **v21 학습·추론 코드와 최종 학습 모델**입니다. 수치 CatBoost와 독립 이미지 시계열 모델을 각각 학습한 뒤, 6단계 확률을 75% / 25%로 결합합니다.

대회 데이터와 이미지, 행별 예측, 실험 ID가 들어 있는 분할 파일, 특징 캐시는 포함하지 않습니다. 데이터는 사용자가 별도로 준비합니다.

## 모델 구성

| 부분 | 구성 |
|---|---|
| 수치 모델 | 인과적 수치 특징 77개, CatBoost 라우터 `1 / (2,3) / (4,5,6)` 및 두 단계 전문가 |
| 이미지 특징 | 고정 ImageNet ResNet18, 전체 화면 및 중앙 75% 영역의 512차원 특징 각각 추출 |
| 이미지 분류 | 현재로부터 120, 90, 60, 45, 30, 20, 10, 0초 기준의 과거 영상 특징을 사용하는 TCN |
| 최종 확률 | `0.75 * numeric_probability + 0.25 * image_probability` |

중앙 영역은 고정 crop입니다. RGB를 실제 픽셀 온도로 복원하지 않습니다. 이력에는 같은 실험의 현재·과거 관측만 사용하며 실험 ID와 정답 label을 모델 특징에서 제외합니다. 누락된 과거 프레임은 관측 가능 여부와 실제 경과 시간으로 처리합니다.

## 학습된 모델로 추론

기존 CatBoost·PyTorch·TorchVision 환경이 있으면 그대로 사용합니다. 새 환경의 의존성 목록은 `requirements.txt`에 있습니다. GPU에서는 해당 장치를 지원하는 PyTorch·TorchVision 조합을 사용하세요.

```bash
git clone https://github.com/x2-qp-cheese/uni_D.git
cd uni_D
D=/path/to/battery_data_light
mkdir -p runs/v21_inference

python -u test.py \
  --data_dir "$D" \
  --model_dir models/v21 \
  --output runs/v21_inference/submission.csv \
  --sample_submission "$D/sample_submission.csv" \
  --image_device cuda:0 --encoder_batch_size 64 --jobs 4
```

CPU 추론에는 `--image_device cpu`를 사용합니다. 저장 모델에는 encoder 가중치도 있어 추론 시 ImageNet 가중치를 새로 다운로드하지 않습니다. 모델 파일의 무결성은 `models/v21/bundle.json`의 SHA256으로 확인합니다.

최종 추론에 필요한 파일은 `models/v21/`의 `router.cbm`, `pair23.cbm`, `late456.cbm`, `image_head.joblib`, `encoder.joblib`, `bundle.json`입니다. 가장 큰 파일은 약 41.7MB이며 개별 파일이 모두 100MB 미만입니다. 중간 checkpoint와 특징 캐시는 배포하지 않습니다.

## 데이터 구조

```text
battery_data_light/
  train/train.csv
  train/images/...
  validation/validation.csv
  validation/images/...
  test/test.csv
  test/images/...
  sample_submission.csv
```

CSV의 `image_path`로 이미지 위치를 찾습니다. 다른 위치에는 `--train_csv`, `--validation_csv`, `--test_csv`, `--image_root`를 사용할 수 있습니다. 원본 CSV 전체를 넣어 단계 선택 전에 수치·영상 이력을 구성합니다.

기본 설정은 이미지 누락·0바이트·손상 시 중단합니다. 불완전한 데이터로 진행하는 경우에만 `--allow_partial_images`를 명시하세요. 내용이 있는 손상 이미지는 이 옵션에서도 중단합니다. 부분 이미지 추론에서 현재 이미지가 없는 행은 수치 단독 확률로 처리합니다.

## 다시 학습하기

```bash
tmux new -s uni_d_v21
# 사용하는 가상환경을 활성화한 상태에서 실행
D=/path/to/battery_data_light
mkdir -p runs/v21_main
set -o pipefail

python -u train.py \
  --data_dir "$D" \
  --output_dir runs/v21_main \
  --task_type GPU --devices 0 --gpu_ram_part 0.60 \
  --image_device cuda:0 --jobs 4 --folds 3 \
  --iterations 600 --image_epochs 12 \
  --encoder_batch_size 64 --image_batch_size 128 \
  --resume \
  2>&1 | tee runs/v21_main/train.log
```

기본 실행은 Train 실험 단위 3-fold 검증과 전체 Train 최종 학습을 수행합니다. 원래 Validation이 있으면 최종 모델을 별도로 평가하며 학습에 합치지 않습니다. 기본 예산은 CatBoost 3개와 이미지 head 1개를 각 fold 및 최종 단계에서 학습하는 총 16회 분류기 적합입니다. ResNet18은 고정합니다.

`--stage cv`는 교차검증만, `--stage final`은 최종 학습과 외부 Validation 평가를 수행합니다. `--no_external_validation`으로 외부 평가를 생략할 수 있습니다. 학습 시 공개 ResNet18 가중치가 캐시에 없으면 다운로드하며, 오프라인 환경에는 `--weights_path /path/to/resnet18-f37072fd.pth`를 사용합니다.

실험 ID가 들어 있는 기존 분할 파일은 공개 저장소에서 제외했습니다. 기본 실행은 Train 안에서 라벨·조건 균형을 고려한 실험 분할을 생성합니다. 기존 비공개 분할을 사용하려면 `--split_manifest /private/path/split_manifest.json`을 지정하세요. 기존 실행 재개에는 당시 코드·설정·분할 파일을 그대로 유지해야 합니다.

`--gpu_ram_part`는 CatBoost에 적용합니다. PyTorch 경로의 메모리는 encoder·image 배치 크기로 조절합니다. 데이터나 저장된 설정을 바꾼 비교는 새로운 output_dir에서 실행합니다.

## 확인된 성능과 한계

전체 데이터 실행의 6단계 macro F1입니다.

| 구성 | Train 33,522행, 78실험의 OOF | 별도 Validation 10,200행, 19실험 |
|---|---:|---:|
| 수치 단독 | 0.898119 | 0.773851 |
| 이미지 단독 | 0.730754 | 0.658387 |
| 정상 이미지 결합 | 0.894451 | 0.770463 |
| 검정 이미지 결합 | 0.897431 | 0.773815 |

현재 v21 고정 이미지 결합은 전체 데이터에서 수치 단독보다 낮았습니다. 검정 이미지 대조보다도 낮아 시각 내용의 일관된 성능 기여를 확인하지 못했습니다. 이 저장소는 해당 실험의 재현용 버전입니다. 앞선 개발 과정의 0.894573은 읽을 수 있던 이미지 8,172행의 OOF이며 위 전체 데이터 점수와 평가 행이 다릅니다.

검정 이미지 대조는 현재와 모든 과거 영상의 내용을 검정 RGB로 바꿉니다. CSV의 `thermal_t` 등 수치 입력은 유지하므로 영상 픽셀 내용의 대조 실험입니다. Test 정답은 포함하지 않으며 로컬에서 Test F1을 계산하지 않습니다.

`cv/comparison.csv`, `cv/report.json`, `external_validation/report.json`에 실행별 점수를 저장합니다. 행별 예측과 로그는 로컬 `runs/` 아래에 생성되며 Git에서 제외합니다.

## 저장소 업로드 규칙

- Public 저장소에 코드와 검토된 최종 추론 모델만 포함합니다.
- `.gitignore`에서 `data/`, `runs/`, CSV·이미지·캐시·중간 checkpoint와 비밀 파일을 제외합니다.
- 100MB 이상인 파일은 이 저장소에 포함하지 않습니다. 현재 배포 가중치는 모두 그보다 작습니다.
- GitHub 토큰은 저장소와 서버 파일에 저장하지 않습니다.
- Git 사용자 설정은 저장소에서 `git config user.name ...`, `git config user.email ...`로 설정하고 `--global`을 사용하지 않습니다.
