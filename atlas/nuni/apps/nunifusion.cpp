// nunifusion — 숨결지기 atlas 네이티브 서비스(com.atlas.NuniFusion1)
//
// 한 프로세스에서 세 모듈을 돌린다(각각 끌 수 있음). 모듈끼리는 Python 분산 모드와 똑같이 MQTT 브로커를 거친다.
//   radar  : ESPHome(mr60bha2/*) 구독 → sensor/radar 1초마다 발행      (radar_bridge.py)
//   cry    : 마이크(PulseAudio) → YAMNet(TFLite) → audio/cry 1초마다 발행  (cry_classifier.py 실물 경로)
//   fusion : sensor/radar · audio/cry · sensor/env 구독 → fusion/state · fusion/alert · control/action 발행
//            (fusion.py + sleep_state.py + env_control.py + personalization.py)
//   + D-Bus: com.atlas.NuniFusion1 이름을 가져가 서비스 매니저에 RUNNING으로 보이고, 상태를 메서드·신호로 내보낸다.
//
// 설정은 환경변수(systemd 유닛의 Environment=)로 하고, 시험할 때는 명령줄 옵션으로 덮어쓴다. --help 참고.
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "nuni/audio_source.hpp"
#include "nuni/cry_detector.hpp"
#include "nuni/fusion.hpp"
#include "nuni/json_util.hpp"
#include "nuni/mqtt.hpp"
#include "nuni/radar_bridge.hpp"
#include "nuni/topics.hpp"
#ifdef NUNI_WITH_TFLITE
#include "nuni/yamnet_tflite.hpp"
#endif
#ifdef NUNI_WITH_DBUS
#include "nuni/dbus_service.hpp"
#endif

using namespace nuni;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

std::string env_str(const char* k, const std::string& def = "") {
  const char* v = std::getenv(k);
  return v ? std::string(v) : def;
}
bool env_on(const char* k, bool def) {
  const char* v = std::getenv(k);
  if (!v) return def;
  return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}
double env_num(const char* k, double def) {
  const char* v = std::getenv(k);
  return v ? std::strtod(v, nullptr) : def;
}

std::string exe_dir() {
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  buf[n] = 0;
  std::string p(buf);
  const auto s = p.rfind('/');
  return s == std::string::npos ? "." : p.substr(0, s);
}

bool file_exists(const std::string& p) { return access(p.c_str(), R_OK) == 0; }

std::string default_model() {
  const std::string emb = exe_dir() + "/yamnet_emb.tflite";
  return file_exists(emb) ? emb : exe_dir() + "/yamnet.tflite";
}

struct Options {
  std::string host = env_str("NUNI_MQTT_HOST", "localhost");
  int port = static_cast<int>(env_num("NUNI_MQTT_PORT", 1883));
  std::string user = env_str("NUNI_MQTT_USER"), pass = env_str("NUNI_MQTT_PASS");
  bool radar = env_on("NUNI_ENABLE_RADAR", true);
  bool cry = env_on("NUNI_ENABLE_CRY", true);
  bool fusion = env_on("NUNI_ENABLE_FUSION", true);
  bool dbus = env_on("NUNI_ENABLE_DBUS", true);
  bool dbus_session = env_str("NUNI_DBUS_BUS", "system") == "session";
  // 기본은 실행 파일 옆 yamnet_emb.tflite(실수형 — atlas에서 노트북과 출력이 같음을 확인), 없으면 yamnet.tflite
  std::string model = env_str("NUNI_YAMNET_MODEL", default_model());
  std::string classmap = env_str("NUNI_YAMNET_CLASSMAP", exe_dir() + "/yamnet_class_map.csv");
  int threads = static_cast<int>(env_num("NUNI_YAMNET_THREADS", 1));
  double cry_threshold = env_num("NUNI_CRY_THRESHOLD", 0.3);  // cry_classifier.py CRY_THRESHOLD
  int mic_sr = static_cast<int>(env_num("NUNI_MIC_SR", 48000));  // USB 마이크 고유 레이트(cry_classifier.py)
  std::string pulse_server = env_str("NUNI_PULSE_SERVER"), pulse_source = env_str("NUNI_PULSE_SOURCE");
  std::string wav = env_str("NUNI_CRY_WAV");  // 지정하면 마이크 대신 이 파일을 반복 재생
};

void usage(const char* argv0) {
  std::cout << "사용법: " << argv0 << " [옵션]\n"
            << "  --host H / --port P     MQTT 브로커 (NUNI_MQTT_HOST, NUNI_MQTT_PORT; 기본 localhost:1883)\n"
            << "  --no-radar --no-cry --no-fusion --no-dbus   모듈 끄기 (NUNI_ENABLE_*=0)\n"
            << "  --session-bus           D-Bus 세션 버스 사용(개발 PC 시험용, NUNI_DBUS_BUS=session)\n"
            << "  --model PATH            YAMNet .tflite (NUNI_YAMNET_MODEL; 기본 실행 파일 옆 yamnet_emb.tflite → yamnet.tflite)\n"
            << "  --classmap PATH         yamnet_class_map.csv (NUNI_YAMNET_CLASSMAP; 없으면 클래스 19·20)\n"
            << "  --wav PATH              마이크 대신 WAV 반복 재생 (NUNI_CRY_WAV)\n"
            << "  --mic-sr N              마이크 샘플레이트 (NUNI_MIC_SR; 기본 48000)\n"
            << "그 밖의 환경변수: NUNI_CRY_THRESHOLD, NUNI_YAMNET_THREADS, NUNI_PULSE_SERVER, NUNI_PULSE_SOURCE,\n"
            << "  NUNI_FUSION_CRY_THRESHOLD, 레이더 게이팅 값(NUNI_BR_MIN 등, radar_bridge.py와 같음)\n";
}

bool parse_args(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    std::string v;
    if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else if (a == "--no-radar") o.radar = false;
    else if (a == "--no-cry") o.cry = false;
    else if (a == "--no-fusion") o.fusion = false;
    else if (a == "--no-dbus") o.dbus = false;
    else if (a == "--session-bus") o.dbus_session = true;
    else if (a == "--host" && next(v)) o.host = v;
    else if (a == "--port" && next(v)) o.port = std::atoi(v.c_str());
    else if (a == "--model" && next(v)) o.model = v;
    else if (a == "--classmap" && next(v)) o.classmap = v;
    else if (a == "--wav" && next(v)) o.wav = v;
    else if (a == "--mic-sr" && next(v)) o.mic_sr = std::atoi(v.c_str());
    else {
      std::cerr << "알 수 없는 옵션: " << a << "\n";
      usage(argv[0]);
      return false;
    }
  }
  return true;
}

// 울음 감지 스레드: 1초마다 최근 1.5초로 판정해 audio/cry를 발행한다.
void cry_loop(const Options& o, MqttClient& mqtt) {
#ifndef NUNI_WITH_TFLITE
  (void)o;
  (void)mqtt;
  std::cerr << "[cry] TFLite 없이 빌드됨 — 울음 감지 끔\n";
#else
  auto model = make_yamnet_tflite(o.model, o.threads);
  if (!model) {
    std::cerr << "[cry] YAMNet 모델을 쓸 수 없어 울음 감지를 끕니다 (" << o.model << ")\n";
    return;
  }
  CryDetector det(std::move(model), file_exists(o.classmap) ? o.classmap : "");
  RingBuffer ring(o.mic_sr, 3.0);
  std::unique_ptr<AudioSource> src = o.wav.empty() ? make_pulse_source(ring, o.pulse_server, o.pulse_source)
                                                   : make_wav_source(ring, o.wav, true);
  bool started = src->start();
  auto last_try = std::chrono::steady_clock::now();
  auto next = std::chrono::steady_clock::now();
  long n = 0;
  bool warned = false;
  while (!g_stop) {
    next += std::chrono::seconds(1);
    std::this_thread::sleep_until(next);
    if (!started || !src->ok()) {  // 마이크가 빠지거나 오디오 서버가 재시작되면 5초마다 다시 연다
      if (!warned) {
        std::cerr << "[cry] 소리 입력 없음(" << src->name() << ") — 5초마다 다시 시도\n";
        warned = true;
      }
      if (std::chrono::steady_clock::now() - last_try > std::chrono::seconds(5)) {
        last_try = std::chrono::steady_clock::now();
        started = src->start();
        if (started) warned = false;
      }
      continue;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto msg = det.classify(ring.last(1.5), o.mic_sr, o.cry_threshold, wall_now());
    if (!msg) continue;
    mqtt.publish(topics::CRY, dump(*msg));
    if (++n == 1 || n % 600 == 0) {
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::cerr << "[cry] " << n << "번째 판정 " << ms << "ms, 점수 " << (*msg)["confidence"] << "\n";
    }
  }
  src->stop();
#endif
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  if (!parse_args(argc, argv, o)) return 2;
  struct sigaction sa {};
  sa.sa_handler = on_signal;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);

  std::cerr << "[nuni] 시작: radar " << o.radar << ", cry " << o.cry << ", fusion " << o.fusion << ", MQTT " << o.host
            << ":" << o.port << "\n";

#ifdef NUNI_WITH_DBUS
  std::unique_ptr<DbusService> dbus;
  if (o.dbus) dbus = make_dbus_service(!o.dbus_session);
#endif

  MqttClient mqtt(o.host, o.port, "nunifusion-" + std::to_string(getpid()));
  if (!o.user.empty()) mqtt.set_credentials(o.user, o.pass);

  long n_radar_out = 0, n_state = 0, n_in = 0;
  bool last_crying = false;

  // fusion이 발행하는 메시지 → MQTT(+ D-Bus 신호)
  Fusion fusion([&](const std::string& topic, const Json& payload) {
    mqtt.publish(topic, dump(payload));
#ifdef NUNI_WITH_DBUS
    if (dbus) {
      if (topic == topics::FUSION_STATE) dbus->emit_state(payload);
      else if (topic == topics::ALERT)
        dbus->emit_alert(payload.value("level", ""), payload.value("reason", ""));
    }
#endif
    if (topic == topics::FUSION_STATE) ++n_state;
  });

  RadarBridge radar;
  if (o.radar)
    for (const auto& t : radar.esphome_topics()) mqtt.subscribe(t);
  if (o.fusion)
    for (const char* t : {topics::RADAR, topics::CRY, topics::ENV}) mqtt.subscribe(t);

  bool warned_bad = false;
  mqtt.set_on_message([&](const std::string& topic, const std::string& raw) {
    ++n_in;
    if (o.radar && topic.rfind(radar.config().prefix + "/", 0) == 0) radar.on_esphome(topic, raw, wall_now());
    if (!o.fusion || !(topic == topics::RADAR || topic == topics::CRY || topic == topics::ENV)) return;
    auto j = parse_payload(raw);
    if (!j || !j->is_object()) {
      // Python 원본은 여기서 예외로 멈춘다. C++은 그 메시지만 버리고 계속 간다.
      if (!warned_bad) std::cerr << "[fusion] JSON 객체가 아닌 메시지를 무시함: " << topic << " " << raw.substr(0, 80) << "\n";
      warned_bad = true;
      return;
    }
#ifdef NUNI_WITH_DBUS
    if (dbus) {
      dbus->set_sensor(topic == topics::RADAR ? "radar" : topic == topics::CRY ? "cry" : "env", *j);
      if (topic == topics::CRY) {
        const bool crying = truthy(pyget(*j, "is_crying"));
        if (crying && !last_crying) dbus->emit_cry(num_or(pyget(*j, "confidence", 0), 0));
        last_crying = crying;
      }
    }
#endif
    (void)last_crying;
    fusion.on_message(topic, *j);
  });

  std::thread cry_th;
  if (o.cry) cry_th = std::thread([&] { cry_loop(o, mqtt); });

  using clock = std::chrono::steady_clock;
  auto next_tick = clock::now() + std::chrono::seconds(1);
  auto next_report = clock::now() + std::chrono::seconds(60);
  bool radar_waiting = false;
  while (!g_stop) {
    const auto now = clock::now();
    const int wait_ms = static_cast<int>(
        std::max<long>(0, std::min<long>(200, std::chrono::duration_cast<std::chrono::milliseconds>(next_tick - now).count())));
    mqtt.poll(wait_ms);
    if (clock::now() >= next_tick) {
      next_tick += std::chrono::seconds(1);
      if (clock::now() - next_tick > std::chrono::seconds(2)) next_tick = clock::now() + std::chrono::seconds(1);
      if (o.radar) {
        if (auto m = radar.step(wall_now())) {
          mqtt.publish(topics::RADAR, dump(*m));
          ++n_radar_out;
          radar_waiting = false;
        } else if (!radar_waiting) {
          std::cerr << "[radar] ESPHome 수신 대기 중 (레이더 미연결/stale)\n";
          radar_waiting = true;
        }
      }
    }
    if (clock::now() >= next_report) {
      next_report += std::chrono::seconds(60);
      std::cerr << "[nuni] 1분 요약: MQTT " << (mqtt.connected() ? "접속" : "끊김") << ", 받은 메시지 " << n_in
                << ", 레이더 발행 " << n_radar_out << ", 상태 발행 " << n_state << "\n";
    }
  }

  std::cerr << "[nuni] 종료 중...\n";
  if (cry_th.joinable()) cry_th.join();
  return 0;
}
