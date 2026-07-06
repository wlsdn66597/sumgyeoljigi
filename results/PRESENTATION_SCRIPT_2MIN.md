# 2분 발표 대본

NUNI는 기존 단순 시뮬레이션을 synthetic raw radar-like signal과 DSP 기반 검증 구조로 확장했습니다. 합성 레이더 신호에서 FFT 기반 BPM 추정을 수행했고, 이번 실행에서 BPM MAE는 0.500 breaths/min, synthetic apnea detection delay는 4.0초였습니다.

레이더 강건성은 SNR 5, 10, 15, 20dB와 window 15, 30, 60초를 조합해 평가했습니다. 이번 실행에서는 15초 window에서 MAE 1.000, 30초와 60초 window에서 MAE 0.000이었고, SNR별 차이는 관측되지 않았습니다.

멀티모달 융합은 11개 controlled scenario에서 검증했습니다. single-sensor baseline은 false alarm 6, miss 0이었고, radar_only는 false alarm 0, miss 2였습니다. full_fusion은 false alarm 0, miss 0으로 경계 호흡 상황을 울음과 환경 신호로 교차검증했습니다.

알림 안정화는 synthetic streaming timeline에서 확인했습니다. 디바운스와 쿨다운을 적용하자 총 알림은 4건에서 1건으로 줄었고, transient false alert는 3건에서 0건으로 줄었습니다. 대신 sustained apnea 감지는 1초 지연됐습니다.

음성 인텐트는 5개 intent와 11개 curated phrase에서 100.0%를 기록했습니다. 다만 실제 STT나 마이크 소음 환경 검증은 아닙니다. 클라우드 리포트는 synthetic night timeline으로 울음 4회, 무호흡 의심 1회, 평균 호흡 39.9회/분의 아침 리포트를 생성했습니다.

울음 이유 분류는 Donate-a-Cry 데이터가 `hungry` 83.6%로 치우쳐 있어 실사용 성능으로 주장하지 않습니다. 현재는 실험적 기능으로 남기고, 실제 하드웨어 통합과 대시보드 캡처, E2E 지연 관측은 2차 데모 과제로 진행합니다.
