// com.atlas.NuniFusion1 D-Bus 서비스 — sdbus-c++ 2.x (atlas SDK에 포함, LG 샘플 lifecycleservice와 같은 방식).
#include "nuni/dbus_service.hpp"

#include <sdbus-c++/sdbus-c++.h>

#include <chrono>
#include <iostream>
#include <mutex>

namespace nuni {

namespace {

constexpr const char* kIface = "com.atlas.NuniFusion1";

class SdbusService : public DbusService {
 public:
  explicit SdbusService(bool system_bus) : start_(std::chrono::steady_clock::now()) {
    // 서비스 매니저는 이 이름(= serviceinfo.json의 id)이 버스에 나타나면 RUNNING으로 본다
    conn_ = system_bus ? sdbus::createSystemBusConnection(sdbus::ServiceName(kDbusName))
                       : sdbus::createSessionBusConnection(sdbus::ServiceName(kDbusName));
    obj_ = sdbus::createObject(*conn_, sdbus::ObjectPath(kDbusPath));
    obj_->addVTable(
            sdbus::registerMethod("Ping").withOutputParamNames("pong").implementedAs([this]() { return ping(); }),
            sdbus::registerMethod("GetState").withOutputParamNames("state_json").implementedAs([this]() {
              std::lock_guard<std::mutex> lk(mu_);
              return state_.is_null() ? std::string("{}") : dump(state_);
            }),
            sdbus::registerMethod("GetSensors").withOutputParamNames("sensors_json").implementedAs([this]() {
              std::lock_guard<std::mutex> lk(mu_);
              return dump(sensors_);
            }),
            sdbus::registerSignal("StateChanged").withParameters<std::string>("state_json"),
            sdbus::registerSignal("Alert").withParameters<std::string, std::string>("level", "reason"),
            sdbus::registerSignal("CryDetected").withParameters<double>("score"))
        .forInterface(kIface);
    conn_->enterEventLoopAsync();
    std::cerr << "[dbus] " << kDbusName << " 등록 (" << (system_bus ? "system" : "session") << " bus)\n";
  }

  ~SdbusService() override {
    conn_->leaveEventLoop();
    obj_.reset();
  }

  void emit_state(const Json& state) override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      state_ = state;
    }
    obj_->emitSignal("StateChanged").onInterface(kIface).withArguments(dump(state));
  }

  void emit_alert(const std::string& level, const std::string& reason) override {
    obj_->emitSignal("Alert").onInterface(kIface).withArguments(level, reason);
  }

  void emit_cry(double score) override { obj_->emitSignal("CryDetected").onInterface(kIface).withArguments(score); }

  void set_sensor(const std::string& key, const Json& msg) override {
    std::lock_guard<std::mutex> lk(mu_);
    sensors_[key] = msg;
  }

 private:
  std::string ping() {
    const auto up = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_);
    return "pong:" + std::to_string(up.count());
  }

  std::unique_ptr<sdbus::IConnection> conn_;
  std::unique_ptr<sdbus::IObject> obj_;
  std::chrono::steady_clock::time_point start_;
  std::mutex mu_;
  Json state_;
  Json sensors_ = Json::object();
};

}  // namespace

std::unique_ptr<DbusService> make_dbus_service(bool system_bus) {
  try {
    return std::make_unique<SdbusService>(system_bus);
  } catch (const sdbus::Error& e) {
    std::cerr << "[dbus] 등록 실패: " << e.getName() << " — " << e.getMessage() << " (D-Bus 없이 계속)\n";
  } catch (const std::exception& e) {
    std::cerr << "[dbus] 등록 실패: " << e.what() << " (D-Bus 없이 계속)\n";
  }
  return nullptr;
}

}  // namespace nuni
