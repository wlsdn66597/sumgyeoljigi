// 멀티모달 융합 판단 엔진 — fusion.py 이식.
//
// decide(): 레이더 + 울음 + 환경을 교차검증하는 규칙(핵심 차별점)
// Fusion  : 최신 신호를 모아 판단하고 디바운스·쿨다운을 거쳐 상태/경보/환경 권고를 발행한다.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nuni/json_util.hpp"
#include "nuni/personalization.hpp"

namespace nuni {

struct Decision {
  std::string level;  // normal / attention / alert
  std::vector<std::string> reasons;
};

// ToF(sensor/tof): 침대 난간 높이에서 수평으로 보는 거리 센서. 빔에 무언가 걸리면 아기가 일어선 것으로 본다.
inline constexpr double STAND_MIN_S = 2.0;    // 이만큼 계속 감지돼야 '일어섬'(손·장난감이 잠깐 스치는 것은 무시)
inline constexpr double STAND_LONG_S = 30.0;  // 이만큼 계속 서 있으면 경보(난간을 넘을 수 있음)
inline constexpr double TOF_STALE_S = 5.0;    // ToF 메시지가 이만큼 안 오면 감지 없음으로 본다(센서 끊김)

// 판단 규칙(상태 없음). radar·cry·env는 마지막 메시지(없으면 null).
// stand_s: ToF가 계속 감지한 시간(초). 없으면 감지 없음(Fusion이 계산해 넘긴다).
Decision decide(const Json& radar, const Json& cry, const Json& env, std::optional<double> stand_s = std::nullopt);

// 비교용: 개별 신호 하나만으로 경보(오탐이 많음).
Decision single_sensor_decide(const Json& radar, const Json& cry, const Json& env);

class Fusion {
 public:
  using PublishFn = std::function<void(const std::string& topic, const Json& payload)>;
  using NowFn = std::function<double()>;

  struct Result {
    std::string level;
    std::vector<std::string> reasons;
    bool alert_emitted = false;
  };

  // publish: 발행 함수(MQTT 등). now: 시계(기본 wall_now). 테스트에서는 가짜 시계를 넣는다.
  explicit Fusion(PublishFn publish, int debounce_n = 2, double cooldown_s = 10, NowFn now = wall_now);

  // sensor/radar · audio/cry · sensor/env · sensor/tof 메시지 하나를 받아 판단·발행한다. 다른 토픽은 무시하고 판단만 다시 한다.
  Result on_message(const std::string& topic, const Json& payload);

  // 한 틱에 모달들을 한꺼번에 갱신하고 1회 판단(시계열 평가용). null이면 갱신하지 않는다.
  Result update(const Json& radar, const Json& cry, const Json& env, const Json& tof = Json());

  // 마지막으로 발행한 fusion/state(아직 없으면 null). D-Bus GetState 등에 쓴다.
  const Json& last_state() const { return last_state_; }

 private:
  std::string debounce(const std::string& level);
  bool cooldown_allows(const std::string& level, const std::vector<std::string>& reasons);
  Result process();
  std::optional<double> stand_seconds();

  PublishFn publish_;
  NowFn now_;
  Json radar_, cry_, env_, tof_;
  std::optional<double> tof_rx_;      // 마지막 ToF 메시지를 받은 시각
  std::optional<double> stand_since_;  // ToF 감지가 이어지기 시작한 시각
  std::string last_ = "normal";
  int debounce_n_;
  double cooldown_s_;
  std::string candidate_ = "normal";
  int candidate_count_ = 0;
  bool has_last_alert_ = false;
  std::string last_alert_key_;
  double last_alert_ts_ = 0.0;
  Personalizer personalizer_;
  Json last_state_;
};

}  // namespace nuni
