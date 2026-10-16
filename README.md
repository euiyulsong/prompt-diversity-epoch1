# 학습 프롬프트 다양성 × 평가 프롬프트 다양성

실행 가능한 실험 키트와 epoch 1, batch size 1 실험 결과를 포함합니다.

## 실행
```bash
unzip prompt_diversity_eval.zip
cd prompt_diversity_eval
bash run.sh
# 간단한 실행 확인 (통계적 결론 용도 아님)
# bash run.sh --train-n 32 --test-n 8 --epochs 1 --seeds 13
```
CUDA GPU와 Hugging Face 모델/데이터 다운로드 연결 필요. 기본 모델 Qwen/Qwen2.5-0.5B-Instruct, LoRA r=16. 0.5B용 16GB급 GPU 권장이나 실제 메모리는 환경에 따라 달라짐. 큰 모델은 --model Qwen/Qwen2.5-1.5B-Instruct로 변경 가능. 전체 기본 실행은 4조건 × 3 seed = 12회 학습 + base 평가. 시간은 GPU에 따라 다름. 의존성 범위는 명시했지만 이 환경에서 모델 다운로드/학습 호환성은 검증하지 못함. 실행 후 pip freeze도 보존.

## 프롬프트와 조건
프롬프트 = 질문 내용이 아닌 instruction 및 필드 배치. 모든 조건에서 동일한 영어 extractive QA와 정답 span 출력 계약 유지. 시스템 문장은 고정, user template만 변형. 원본 SQuAD 질문을 paraphrase하는 실험과는 구분.

| 학습 조건 | 학습 template ID | 단일 평가 0 | 다양 평가 1~4 | 미관측 다양 평가 5~8 |
|---|---|---|---|---|
| base | 학습 없음 | 평가 | 평가 | 평가 |
| single_target | 0 | 동일 template | 다른 template | 다른 template |
| single_other | 1 | 다른 template | 일부 동일 | 다른 template |
| diverse_other | 1~4 균등 | 다른 template | 동일 pool | 다른 pool |
| diverse_including_target | 0~3 균등 | 일부 동일 | 일부 동일 | 다른 pool |

※ `diverse_including_target`의 학습 pool `0~3`은 template ID 0, 1, 2, 3을 뜻합니다. target인 0을 포함하고 seen 평가 pool(1~4)과는 1~3이 겹치며, template 4는 학습에서 사용하지 않습니다. 반면 `diverse_other`는 1~4로 학습하며 target 0은 제외합니다.

요청 1: 다양한 프롬프트 테스트에서 diverse_other와 single_other 비교. 평가 1~4는 seen-pool, 5~8은 unseen-pool 결과로 별도 해석.
요청 2: 단일 프롬프트 테스트(0)에서 diverse_other와 single_target 비교. single_other와 diverse_including_target을 함께 봐야 다양성 효과와 template 일치 효과를 구분 가능.

## 공정성
- SQuAD train에서 2,000개, 공식 validation에서 100개. 데이터 선택 seed=2026 고정. 학습/평가 간 같은 question 또는 context를 추가 제외.
- 같은 QA, 순서, epoch, batch, optimizer 설정, 모델 초기 가중치. 조건별 프롬프트만 변경. 각 QA당 하나의 template만 배정해 데이터 수 증가 효과 제거. 배정은 seed별 고정하고 epoch마다 바꾸지 않음.
- 모든 template에서 길이 제한을 만족하는 QA만 선택하며 truncation 없음. 선택 편향 때문에 전체 SQuAD 대표 성능으로 해석하지 말 것.
- prompt loss=-100, answer+EOS에만 loss. 출력 형식은 동일.
- 동일 질문 100개 각각을 9개 template 모두로 평가. 테스트 프롬프트가 다양해져도 질문 난이도 동일.
- 예제 수, answer token 수, optimizer step 동일. prompt 길이가 달라 입력 token 수/FLOPs는 완전히 같지 않음.

## 결과
results/metrics.csv: 조건 × seed × 평가군 및 각 template의 EM/F1.
results/summary.json: seed 평균 및 EM seed 표준편차.
results/pred_*.json: 질문 ID, template ID, 예측, 정답, EM/F1.
results/config.json 및 data.json: 설정/선택 데이터와 ID, 프롬프트.
results/<condition>_<seed>: LoRA adapter. base model을 함께 불러와 사용.

### 빠른 실행 결과

RTX 4090에서 `Qwen/Qwen2.5-0.5B-Instruct`, train 2,000개, test 100개, epoch 1, batch size 1, seed 13으로 실행했습니다. 표의 EM/F1은 각 평가군에 포함된 100개 질문 및 prompt template에 대한 평균입니다.

- **단일 평가 (template 0):** 학습 prompt 다양성 효과와 target prompt 일치 효과를 보기 위한 기준 template 하나로 평가합니다.
- **다양 평가 (templates 1–4):** 학습에서 사용한 prompt pool과 겹치거나 같은 pool의 다양한 template로 평가합니다. 특히 `diverse_other`는 이 pool로 학습하므로 seen-pool 일반화를 비교합니다.
- **미관측 다양 평가 (templates 5–8):** 해당 fine-tuning 조건에서 학습에 사용하지 않은 별도 prompt pool로 평가합니다. 여기서 ‘미관측’은 fine-tuning 중 미사용을 뜻하며, 모델 사전학습에서도 보지 않았다는 의미는 아닙니다.

각 평가군은 동일한 100개 질문에 적용되며, 여러 template이 있는 평가군은 질문별 template 점수를 평균합니다.

| 학습 조건 | 단일 평가 (0) EM / F1 | 다양 평가 (1–4) EM / F1 | 미관측 다양 평가 (5–8) EM / F1 |
|---|---:|---:|---:|
| base (학습 없음) | 0.520 / 0.676 | 0.345 / 0.528 | 0.233 / 0.416 |
| single_target | 0.720 / 0.819 | 0.663 / 0.772 | 0.510 / 0.614 |
| single_other | 0.690 / 0.804 | 0.678 / 0.799 | 0.500 / 0.610 |
| diverse_other | 0.690 / 0.787 | 0.690 / 0.800 | 0.515 / 0.617 |
| diverse_including_target | 0.710 / 0.814 | 0.710 / 0.817 | 0.545 / 0.643 |

이 결과는 단일 seed의 탐색적 실행이며, 통계적 결론이나 조건 간 유의성 검정을 제공하지 않습니다. 자세한 지표는 [`results/metrics.csv`](results/metrics.csv), 질문 단위 예측은 `results/pred_*.json`, 설정 및 결과 전체는 [`results/summary.json`](results/summary.json)과 `results/config.json`에 있습니다.

EM은 SQuAD식 소문자/구두점/관사/공백 정규화, 정답 alias 중 최고 점수. token F1도 기록. 설명을 붙이면 EM이 낮아지는 것이 의도된 출력 계약. 평가군 EM CI는 질문 단위 cluster bootstrap으로 계산하며 seed별 CI임. seed std는 학습 변동성이고 CI와 다름. 조건 간 유의성 검정은 포함하지 않음. 100문항은 탐색용이라 작은 차이에 결론 내리지 말 것. 최종 비교는 --test-n 500 이상 및 여러 단일 template/새 prompt pool 반복을 권장.

미관측은 '이 fine-tuning에서 사용하지 않은 template'를 뜻하며 모델 사전학습에서 보지 않았다는 뜻은 아님. 사전학습 데이터 오염 가능성도 남아 있음. 이 실험은 단일 task 영어 extractive QA의 표면적 instruction 다양성만 측정. 한국어/생성형 답변/멀티태스크로 바로 일반화하지 말 것.

## 확인한 범위
위의 epoch 1 실험은 CUDA GPU에서 실제 학습과 평가를 실행했습니다. Python 문법과 pure-Python metric 함수/프롬프트 구성도 확인했습니다.

참고: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
https://huggingface.co/docs/peft/package_reference/lora
https://huggingface.co/datasets/rajpurkar/squad
