# atlas(LG AI Native OS) 이식

## yamnet_test — YAMNet 추론 첫 확인 (2026-10-19)

| 프로그램 | 방식 | 결과 (atlas Pi 4) |
|---|---|---|
| `yamnet_tflite_test` | TensorFlow Lite 직접 사용(XNNPACK, CPU) — LG 가이드 tflite-examples와 같은 방식 | ✅ 노트북 TFLite와 출력이 같다(울음 점수 0.5399, 임베딩 노름 11.7879). 0.975초 창 1개당 49ms(스레드 1) / 24ms(스레드 4) |
| `yamnet_ecoai_test` | ECO.AI 통합 API(ecoai-inference) | ❌ `/usr/lib/ecoai/strategy_cfg` 없음 → `.tflite` 플러그인 미등록. 추론 플러그인 애드온이 기기 이미지와 SDK 모두에 없다(SDK의 /usr/lib/ecoai에는 TTS 플러그인만) |

빌드(개발 컨테이너 안): `cd yamnet_test && mkdir build && cd build && cmake .. && make`
기기 실행: 실행 파일·`yamnet_emb.tflite`·`lib/`(libtensorflowlite.so, libtensorflowlite_xnnpack_delegate.so, libecoai-inference.so.1, libglog.so.1 — 기기에 없음)를 `/mnt/rawdata/nuni`에 두고
`LD_LIBRARY_PATH=$PWD/lib ./yamnet_tflite_test yamnet_emb.tflite [input_15600.f32]`. 시스템 경로는 건드리지 않는다.
