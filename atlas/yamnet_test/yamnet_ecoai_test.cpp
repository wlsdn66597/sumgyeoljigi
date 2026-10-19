// YAMNet on ATLAS ECO.AI (ecoai-inference) — 첫 동작 확인용.
//
// 0.975초(16kHz 15600샘플) float32 원시 파일을 넣어 YAMNet(점수 521 + 임베딩 1024 두 출력 tflite)을 돌리고,
// 출력 텐서 정보, 울음 점수(클래스 19 "Crying, sobbing" / 20 "Baby cry, infant cry"), 상위 5 클래스,
// 임베딩 앞 5개·노름, 평균 추론 시간을 출력한다. 노트북 TFLite 결과와 같은지 비교하는 데 쓴다.
//
// 사용법: yamnet_ecoai_test <yamnet_emb.tflite> [input_15600.f32] [--runs N] [--threads N] [--backend cpu]
//         입력 파일을 주지 않으면 450Hz 사인파(울음이 아닌 소리)를 넣는다.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <ecoai-inference/EcoTypes.hpp>
#include <ecoai-inference/InferenceStrategy/InferenceStrategy.hpp>
#include <ecoai-inference/InferenceStrategy/InferenceStrategyBuilder.hpp>

namespace inf = ecoai::inference;

static constexpr int kWin = 15600;

static std::vector<float> LoadWindow(const std::string& path) {
  std::vector<float> x(kWin, 0.f);
  if (path.empty()) {
    for (int i = 0; i < kWin; ++i) x[i] = 0.3f * std::sin(2.0 * M_PI * 450.0 * i / 16000.0);
    return x;
  }
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "입력 파일을 열 수 없음: " << path << "\n";
    std::exit(1);
  }
  f.read(reinterpret_cast<char*>(x.data()), kWin * sizeof(float));
  return x;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "사용법: " << argv[0] << " <yamnet_emb.tflite> [input_15600.f32] [--runs N] [--threads N] [--backend cpu]\n";
    return 2;
  }
  std::string model = argv[1], input_path, backend = "cpu";
  int runs = 20, threads = 1;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) backend = argv[++i];
    else input_path = argv[i];
  }

  std::unordered_map<std::string, std::string> options;
  options[backend] = "";
  options[backend + ".intra_op_num_threads"] = std::to_string(threads);
  inf::StrategyPtr strategy;
  try {
    strategy = inf::InferenceStrategyBuilder(model).setOptions(options).build();
  } catch (const std::exception& e) {
    std::cerr << "strategy 생성 실패: " << e.what() << "\n";
    return 1;
  }
  if (!strategy->initialize()) {
    std::cerr << "initialize 실패\n";
    return 1;
  }

  auto inputs = strategy->getInputTensorsInfo();
  if (inputs.empty()) {
    std::cerr << "입력 텐서 없음\n";
    return 1;
  }
  inf::EcoTensorInfo in_info = inputs[0];
  std::cout << "입력: " << in_info.name << " dims=[";
  for (size_t i = 0; i < in_info.dimensions.size(); ++i) std::cout << (i ? "," : "") << in_info.dimensions[i];
  std::cout << "] float32=" << (in_info.dataType == inf::EcoTensorInfo::DataType::FLOAT32) << "\n";

  std::vector<float> x = LoadWindow(input_path);
  inf::EcoTensorMap in_map, out_map;
  inf::EcoTensorData td;
  td.info = in_info;
  td.data = x.data();
  td.size = x.size() * sizeof(float);
  td.owns_data = false;
  in_map[in_info.name] = td;

  // 첫 실행(준비) 후 시간 측정
  if (!strategy->execute(in_map, out_map)) {
    std::cerr << "execute 실패\n";
    return 1;
  }
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < runs; ++r) strategy->execute(in_map, out_map);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;

  const float* scores = nullptr;
  const float* emb = nullptr;
  for (auto& kv : out_map) {
    size_t n = kv.second.size / sizeof(float);
    std::cout << "출력: " << kv.first << " 원소 " << n << "\n";
    if (n % 521 == 0 && n >= 521) scores = static_cast<const float*>(kv.second.data);
    if (n % 1024 == 0 && n >= 1024) emb = static_cast<const float*>(kv.second.data);
  }
  if (!scores || !emb) {
    std::cerr << "점수(521)·임베딩(1024) 출력을 찾지 못함\n";
    return 1;
  }
  std::vector<int> idx(521);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  double norm = 0;
  for (int i = 0; i < 1024; ++i) norm += double(emb[i]) * emb[i];
  std::cout.setf(std::ios::fixed);
  std::cout.precision(4);
  std::cout << "울음 점수 " << std::max(scores[19], scores[20]) << " | 상위 클래스";
  for (int k = 0; k < 5; ++k) std::cout << " " << idx[k];
  std::cout << " | 임베딩 앞 5개";
  for (int i = 0; i < 5; ++i) std::cout << " " << emb[i];
  std::cout << " | 노름 " << std::sqrt(norm) << "\n";
  std::cout.precision(2);
  std::cout << "평균 추론 시간 " << ms << " ms (" << runs << "회, 스레드 " << threads << ", 백엔드 " << backend << ")\n";
  return 0;
}
