"""상시 엣지 노드 — 대시보드와 독립적으로 센서·울음·융합을 구동한다.

대시보드가 열려 있지 않아도 모니터링·경보·환경권고가 24시간 동작한다.
분산 실행: NUNI_BUS=mqtt 로 브로커에 발행하면, 대시보드(NUNI_DASHBOARD_ONLY=1)는
브로커 메시지만 구독해 화면을 채운다. systemd 서비스(nuni-edge)로 상시 가동한다.

실행: NUNI_BUS=mqtt python edge_node.py
"""
import time

import workers


def main():
    workers.ensure_started(pipeline=True)   # store 구독 + 센서/울음/융합 파이프라인
    print("[nuni-edge] pipeline started (sensors + cry + fusion). Ctrl+C to stop.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
