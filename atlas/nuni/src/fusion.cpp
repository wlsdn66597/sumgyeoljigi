#include "nuni/fusion.hpp"

#include "nuni/env_control.hpp"
#include "nuni/sleep_state.hpp"
#include "nuni/topics.hpp"

namespace nuni {

namespace {

int order(const std::string& level) {
  if (level == "alert") return 2;
  if (level == "attention") return 1;
  return 0;
}

std::string max_level(const std::string& a, const std::string& b) { return order(a) >= order(b) ? a : b; }

std::string join(const std::vector<std::string>& v, const char* sep) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) out += sep;
    out += v[i];
  }
  return out;
}

}  // namespace

Decision decide(const Json& radar, const Json& cry, const Json& env) {
  std::string level = "normal";
  std::vector<std::string> reasons;
  const bool has_radar = truthy(radar);
  const bool has_cry = truthy(cry);
  const bool has_env = truthy(env);

  const bool apnea = has_radar && truthy(pyget(radar, "apnea"));
  const double br = has_radar ? num_or(pyget(radar, "breathing_rate", pyget(radar, "breathing_bpm", 99)), 99) : 99;
  // 실물 레이더는 측정 조건이 나쁘면 호흡을 못 재고 0을 준다. 이때의 0은 '무호흡'이 아니라 '미측정'이므로
  // 호흡 판정을 보류한다. 명시적 apnea 플래그는 브리지가 조건을 확인하고 세운 것이라 그대로 신뢰한다.
  const bool br_valid = has_radar ? truthy(pyget(radar, "breath_valid", true)) : true;
  if (has_radar && truthy(pyget(radar, "presence")) && (apnea || (br_valid && br < 8))) {
    level = max_level(level, "alert");
    reasons.emplace_back("호흡 미검출·무호흡 의심");
  }

  const double cry_conf = has_cry ? num_or(pyget(cry, "confidence", pyget(cry, "cry_score", 0)), 0) : 0;
  const bool cry_flag = has_cry && truthy(pyget(cry, "is_crying")) && cry_conf >= 0.7;
  if (cry_flag) {
    level = max_level(level, "attention");
    reasons.emplace_back("울음 감지(" + py_str(pyget(cry, "cls")) + ")");
  }

  // 교차검증: 울음 + (호흡 이상 or 큰 움직임)이 동시일 때만 경보로 상향
  // (원본과 같게 br_valid를 보지 않는다 — README '원본에서 발견한 점' 참고)
  if (cry_flag && has_radar && (num_or(pyget(radar, "movement", 0), 0) > 0.6 || apnea || br < 8)) {
    level = max_level(level, "alert");
    reasons.emplace_back("다중 신호 동시 이상");
  }

  // 경계 신호 교차검증: 단독으론 임계 미만인 '경계 호흡'이 다른 모달과 겹치면 경보로 상향.
  const bool irregular = has_radar && truthy(pyget(radar, "presence")) && !apnea && br_valid &&
                         ((8 <= br && br <= 15) || br > 55);
  const bool bad_env = has_env && num_or(pyget(env, "co2", 0), 0) > 1000;
  if (irregular && (cry_flag || bad_env)) {
    level = max_level(level, "alert");
    reasons.emplace_back("경계 호흡+보강신호 교차검증");
  }

  if (has_env) {
    if (num_or(pyget(env, "co2", 0), 0) > 1000) {
      level = max_level(level, "attention");
      reasons.emplace_back("CO₂ 높음");
    }
    const auto t = as_number(pyget(env, "temp"));
    if (t && (*t < 18 || *t > 26)) {
      level = max_level(level, "attention");
      reasons.emplace_back("실내 온도 이상");
    }
  }
  return {level, reasons};
}

Decision single_sensor_decide(const Json& radar, const Json& cry, const Json& env) {
  if (truthy(cry) && truthy(pyget(cry, "is_crying"))) return {"alert", {"울음(단일)"}};
  if (truthy(radar) && truthy(pyget(radar, "apnea"))) return {"alert", {"무호흡(단일)"}};
  if (truthy(radar) && num_or(pyget(radar, "movement", 0), 0) > 0.5) return {"alert", {"움직임(단일)"}};
  if (truthy(env) && num_or(pyget(env, "co2", 0), 0) > 800) return {"alert", {"CO₂(단일)"}};
  return {"normal", {}};
}

Fusion::Fusion(PublishFn publish, int debounce_n, double cooldown_s, NowFn now)
    : publish_(std::move(publish)), now_(std::move(now)), debounce_n_(debounce_n), cooldown_s_(cooldown_s) {}

std::string Fusion::debounce(const std::string& level) {
  if (level == "normal") {
    candidate_ = "normal";
    candidate_count_ = 0;
    return level;
  }
  if (level == candidate_) {
    candidate_count_ += 1;
  } else {
    candidate_ = level;
    candidate_count_ = 1;
  }
  return candidate_count_ >= debounce_n_ ? level : "normal";
}

bool Fusion::cooldown_allows(const std::string& level, const std::vector<std::string>& reasons) {
  const std::string key = level + '\x1f' + join(reasons, "|");
  const double now = now_();
  if (has_last_alert_ && key == last_alert_key_ && now - last_alert_ts_ < cooldown_s_) return false;
  has_last_alert_ = true;
  last_alert_key_ = key;
  last_alert_ts_ = now;
  return true;
}

Fusion::Result Fusion::process() {
  const Decision d = decide(radar_, cry_, env_);
  const std::string level = debounce(d.level);
  const std::vector<std::string> out_reasons = level != "normal" ? d.reasons : std::vector<std::string>{};

  // 일상 모니터링(헤드라인): 수면상태 · 개인화 · 선제 환경제어
  const auto ss = sleep_state::classify(radar_, cry_);
  Json personal = Json::object();
  if (truthy(radar_)) {
    personalizer_.update(as_number(pyget(radar_, "breathing_rate", 0)), as_number(pyget(radar_, "movement", 0.0)));
    personal = personalizer_.summary();
  }
  const auto actions = env_control::recommend(env_, ss.state);
  if (!actions.empty()) {
    Json names = Json::array();
    for (const auto& a : actions) names.push_back(a.first);
    publish_(topics::CONTROL, Json{{"actions", names}, {"ts", now_()}});
  }

  // 대시보드가 구독만으로 전체 상태를 복원하도록 통합 상태를 발행한다.
  Json acts = Json::array();
  for (const auto& a : actions) acts.push_back(Json::array({a.first, a.second}));
  Json state;
  state["level"] = level;
  state["reasons"] = out_reasons;
  state["raw_level"] = d.level;
  state["sleep"] = Json{{"state", ss.state}, {"reason", ss.reason}};
  state["personal"] = personal;
  state["actions"] = acts;
  state["ts"] = now_();
  last_state_ = state;
  publish_(topics::FUSION_STATE, state);

  bool emitted = false;
  if (level != "normal" && level != last_ && cooldown_allows(level, out_reasons)) {
    publish_(topics::ALERT, Json{{"level", level}, {"reason", join(out_reasons, ", ")}, {"ts", now_()}});
    emitted = true;
  }
  last_ = level;
  return {level, out_reasons, emitted};
}

Fusion::Result Fusion::on_message(const std::string& topic, const Json& payload) {
  if (topic == topics::RADAR) radar_ = payload;
  else if (topic == topics::CRY) cry_ = payload;
  else if (topic == topics::ENV) env_ = payload;
  return process();
}

Fusion::Result Fusion::update(const Json& radar, const Json& cry, const Json& env) {
  if (!radar.is_null()) radar_ = radar;
  if (!cry.is_null()) cry_ = cry;
  if (!env.is_null()) env_ = env;
  return process();
}

}  // namespace nuni
