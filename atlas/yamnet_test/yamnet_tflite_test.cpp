// YAMNet on ATLAS — TensorFlow Lite 런타임 직접 사용(XNNPACK, CPU).
//
// ECO.AI 통합 API(ecoai-inference)는 추론 플러그인(애드온 패키지)이 기기·SDK에 없어 쓸 수 없을 때를 위한 경로다.
// LG 가이드의 tflite-examples(mv2_tflite_xnnpack_detector)와 같은 방식이다.
// 출력·사용법은 yamnet_ecoai_test와 같다.
//
// 사용법: yamnet_tflite_test <yamnet_emb.tflite> [input_15600.f32] [--runs N] [--threads N]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

static constexpr int kWin = 15600;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "사용법: " << argv[0] << " <yamnet_emb.tflite> [input_15600.f32] [--runs N] [--threads N]\n";
    return 2;
  }
  std::string model_path = argv[1], input_path;
  int runs = 20, threads = 1;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
    else input_path = argv[i];
  }

  std::vector<float> x(kWin, 0.f);
  if (input_path.empty()) {
    for (int i = 0; i < kWin; ++i) x[i] = 0.3f * std::sin(2.0 * M_PI * 450.0 * i / 16000.0);
  } else {
    std::ifstream f(input_path, std::ios::binary);
    if (!f) { std::cerr << "입력 파일을 열 수 없음\n"; return 1; }
    f.read(reinterpret_cast<char*>(x.data()), kWin * sizeof(float));
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model) { std::cerr << "모델 읽기 실패\n"; return 1; }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interp;
  if (tflite::InterpreterBuilder(*model, resolver)(&interp) != kTfLiteOk || !interp) {
    std::cerr << "인터프리터 생성 실패\n";
    return 1;
  }
  TfLiteXNNPackDelegateOptions opts = TfLiteXNNPackDelegateOptionsDefault();
  opts.num_threads = threads;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)> xnn(TfLiteXNNPackDelegateCreate(&opts),
                                                                               TfLiteXNNPackDelegateDelete);
  if (interp->ModifyGraphWithDelegate(xnn.get()) != kTfLiteOk) std::cerr << "(XNNPACK 위임 실패 — 기본 CPU로 진행)\n";
  interp->SetNumThreads(threads);
  if (interp->AllocateTensors() != kTfLiteOk) { std::cerr << "AllocateTensors 실패\n"; return 1; }

  TfLiteTensor* in = interp->input_tensor(0);
  std::cout << "입력: " << in->name << " 원소 " << in->bytes / sizeof(float) << "\n";
  std::memcpy(interp->typed_input_tensor<float>(0), x.data(), kWin * sizeof(float));
  if (interp->Invoke() != kTfLiteOk) { std::cerr << "Invoke 실패\n"; return 1; }
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < runs; ++r) interp->Invoke();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;

  const float* scores = nullptr;
  const float* emb = nullptr;
  for (size_t o = 0; o < interp->outputs().size(); ++o) {
    const TfLiteTensor* t = interp->output_tensor(o);
    size_t n = t->bytes / sizeof(float);
    std::cout << "출력: " << t->name << " 원소 " << n << "\n";
    if (n == 521) scores = interp->typed_output_tensor<float>(o);
    if (n == 1024) emb = interp->typed_output_tensor<float>(o);
  }
  if (!scores || !emb) { std::cerr << "점수·임베딩 출력을 찾지 못함\n"; return 1; }
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
  std::cout << "평균 추론 시간 " << ms << " ms (" << runs << "회, 스레드 " << threads << ")\n";
  return 0;
}
