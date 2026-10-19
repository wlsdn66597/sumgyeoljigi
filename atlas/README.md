# atlas(LG AI Native OS) 이식

## yamnet_test — YAMNet 추론 첫 확인 (2026-10-19)

| 프로그램 | 방식 | 결과 (atlas Pi 4) |
|---|---|---|
| `yamnet_tflite_test` | TensorFlow Lite 직접 사용(XNNPACK, CPU) — LG 가이드 tflite-examples와 같은 방식 | ✅ 노트북 TFLite와 출력이 같다(울음 점수 0.5399, 임베딩 노름 11.7879). 0.975초 창 1개당 49ms(스레드 1) / 24ms(스레드 4) |
| `yamnet_ecoai_test` | ECO.AI 통합 API(ecoai-inference) | ❌ `/usr/lib/ecoai/strategy_cfg` 없음 → `.tflite` 플러그인 미등록. 추론 플러그인 애드온이 기기 이미지와 SDK 모두에 없다(SDK의 /usr/lib/ecoai에는 TTS 플러그인만) |

### atlas Pi 5 (ATLAS 26.06.0-246, 8GB) — ECO.AI 동작 확인

Pi 5 이미지에는 추론 플러그인이 들어 있다: `/usr/lib/ecoai/strategy_cfg`에 TfLiteStrategy(cpu·gpu), OnnxStrategy(cpu), ExecuTorchStrategy(cpu)가 있고, `libecoai-llm·voice·vision·vectordb`, TFLite·ONNX Runtime도 시스템에 있다(`lib/` 불필요).

| 프로그램 | 결과 (atlas Pi 5) |
|---|---|
| `yamnet_ecoai_test` (ECO.AI, cpu) | ✅ 출력 동일(0.5399 / 11.7879), 8.8ms |
| `yamnet_ecoai_test --backend gpu` | ✅ 출력 동일, 17.5ms (CPU보다 느림 → cpu 사용) |
| `yamnet_tflite_test` | ✅ 8.8ms(스레드 1) / 5.9ms(스레드 4) |

빌드(개발 컨테이너 안): `cd yamnet_test && mkdir build && cd build && cmake .. && make`
기기 실행: 실행 파일·`yamnet_emb.tflite`·`lib/`(libtensorflowlite.so, libtensorflowlite_xnnpack_delegate.so, libecoai-inference.so.1, libglog.so.1 — 기기에 없음)를 `/mnt/rawdata/nuni`에 두고
`LD_LIBRARY_PATH=$PWD/lib ./yamnet_tflite_test yamnet_emb.tflite [input_15600.f32]`. 시스템 경로는 건드리지 않는다.

## nuni — 숨결지기 서비스 C++ 이식 (2026-10-19)

`nuni/`는 파이에서 Python으로 돌던 실시간 경로(레이더 변환·울음 감지·융합 판단)를 C++ 네이티브 서비스 `com.atlas.NuniFusion1`로 옮긴 것이다. 울음 감지는 위 `yamnet_tflite_test`와 같은 방식(TFLite 직접 사용, XNNPACK)이다. 같은 입력을 넣었을 때 Python 원본과 출력이 완전히 같은지 비교 시험을 붙여 두었다. 빌드·배포·검증 결과는 [nuni/README.md](nuni/README.md)에 있다.
