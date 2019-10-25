// Benchmarking inferences on co-compiled models vs stand alone edgetpu models.
// If the EdgeTpuManager detects 2 different TPUs, this program will isolate the "control"
// models and the co-compile models into different TPUs, if it's a single TPUs, it switches
// inferencing on a single TPU
/*
 Example usage:
 1. Create directory: 
    mkdir -p /tmp/test_data
 2. wget -O /tmp/test_data/inat_bird_edgetpu.tflite \
     https://github.com/google-coral/edgetpu/blob/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite?raw=true
 3. wget -O /tmp/test_data/inat_plant_edgetpu.tflite \
     https://github.com/google-coral/edgetpu/blob/master/test_data/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite?raw=true
 4. wget -O /tmp/test_data/bird.jpg \
     https://farm3.staticflickr.com/8008/7523974676_40bbeef7e3_o.jpg
 5. wget -O /tmp/test_data/plant.jpg \
     https://c2.staticflickr.com/1/62/184682050_db90d84573_o.jpg
 6. wget -O /tmp/test_data/co_inat_bird.tflite \
     https://github.com/google-coral/edgetpu/blob/master/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite?raw=true
 7. wget -O /tmp/test_data/co_inat_plant.tflite \
     https://github.com/google-coral/edgetpu/blob/master/test_data/mobilenet_v2_1.0_224_inat_plant_quant.tflite?raw=true
 8. cocompile the too tflite models: 
    cd /tmp/test_data && edgetpu_compiler co_inat_bird.tflite co_inat_plant.tflite
 9. cd /tmp/test_data && mogrify -format bmp *.jpg
 10. Build and run `co_compiled_models_benchmark`
*/
// To reduce variation between different runs, one can disable CPU scaling with
//   sudo cpupower frequency-set --governor performance
#include <algorithm>
#include <chrono>  // NOLINT
#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "edgetpu.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

void run(const edgetpu::EdgeTpuManager::DeviceEnumerationRecord& tpu, 
		const std::string& bird_model_path, const std::string& plant_model_path, 
		const std::string& bird_image_path, const std::string& plant_image_path, 
		const int num_inferences, const std::string& model_type, const int batch_size) {
  
  // read in models
  std::unique_ptr<tflite::FlatBufferModel> bird_model =
      tflite::FlatBufferModel::BuildFromFile(bird_model_path.c_str());
  if (bird_model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << bird_model_path
              << std::endl;
    std::abort();
  }
  std::unique_ptr<tflite::FlatBufferModel> plant_model =
      tflite::FlatBufferModel::BuildFromFile(plant_model_path.c_str());
  if (plant_model == nullptr) {
    std::cerr << "Fail to build FlatBufferModel from file: " << plant_model_path
              << std::endl;
    std::abort();
  }

  // This context is shared among multiple models.
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = 
	  edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            tpu.type, tpu.path);
  std::unique_ptr<tflite::Interpreter> bird_interpreter =
      coral::BuildEdgeTpuInterpreter(*bird_model, edgetpu_context.get());
  std::unique_ptr<tflite::Interpreter> plant_interpreter =
      coral::BuildEdgeTpuInterpreter(*plant_model, edgetpu_context.get());

  // processing images
  std::vector<uint8_t> bird_input = coral::GetInputFromImage(
      bird_image_path, coral::GetInputShape(*bird_interpreter, 0));
  std::vector<uint8_t> plant_input = coral::GetInputFromImage( 
      plant_image_path, coral::GetInputShape(*plant_interpreter, 0));

  // start timer
  const auto& start_time = std::chrono::steady_clock::now();

  if (batch_size) {
    int num_iterations = (num_inferences + batch_size - 1) / batch_size;
    std::cout << "batchsize: " << batch_size
	      << " iterations: " << num_iterations << "\n";
    // Run inference in batch_size alternately
    for (int i = 0; i < num_iterations; ++i) {
      for (int j = 0; j < batch_size; ++j) {
        coral::RunInference(bird_input, bird_interpreter.get());
      }
      for (int j = 0; j < batch_size; ++j) {
        coral::RunInference(plant_input, plant_interpreter.get());
      }
    }
  } else {
    // Run inference alternately
    for (int i = 0; i < num_inferences; i++) {
      coral::RunInference(bird_input, bird_interpreter.get());
      coral::RunInference(plant_input, plant_interpreter.get());
    }
  }

  std::chrono::duration<double> time_span =
      std::chrono::steady_clock::now() - start_time;

  std::cout << "Running " << model_type
	    << " models with " << num_inferences << " inferences" 
	    << " costs: " << time_span.count() << " seconds." << std::endl;
}


ABSL_FLAG(int, batch_size, 0, "Batch size to run models alternately.");

ABSL_FLAG(std::string, bird_model_path,
          "/tmp/test_data/inat_bird_edgetpu.tflite",
          "Path to the provided edgetpu bird tflite model.");

ABSL_FLAG(std::string, plant_model_path,
          "/tmp/test_data/inat_plant_edgetpu.tflite",
          "Path to the downloaded edgetpu plant tflite model.");

ABSL_FLAG(std::string, co_compiled_bird_model_path,
          "/tmp/test_data/co_inat_bird_edgetpu.tflite",
          "Path to the co-compiled bird model.");

ABSL_FLAG(std::string, co_compiled_plant_model_path,
          "/tmp/test_data/co_inat_plant_edgetpu.tflite",
          "Path to the co-compiled plant model.");

ABSL_FLAG(std::string, bird_image_path, "/tmp/test_data/bird.bmp",
          "Path to bird image.");

ABSL_FLAG(std::string, plant_image_path, "/tmp/test_data/plant.bmp",
          "Path to the plant image.");


int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const auto& available_tpus =
      edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();

  if (available_tpus.size() < 1) {
    std::cout << "This benchmark requires at least 1 EdgeTPU\n";
    exit(0);
  }

  std::cout << "Running benchmark with " << available_tpus.size() << " EdgeTPUs"
            << "\n  bird model: " << absl::GetFlag(FLAGS_bird_model_path) 
	    << "\n  plant model: " << absl::GetFlag(FLAGS_plant_model_path)
	    << "\n  co-compiled bird model: " << absl::GetFlag(FLAGS_co_compiled_bird_model_path)
	    << "\n  co-compiled plant model: " << absl::GetFlag(FLAGS_co_compiled_plant_model_path)
	    << "\n";

  for (int num_inferences : {100, 200, 400, 800, 1600, 3200, 6400}) {
    std::cout << "--------------------------------------------------\n";

    run(available_tpus[0], 
                    absl::GetFlag(FLAGS_bird_model_path),
                    absl::GetFlag(FLAGS_plant_model_path),
                    absl::GetFlag(FLAGS_bird_image_path),
                    absl::GetFlag(FLAGS_plant_image_path),
                    num_inferences, "control", absl::GetFlag(FLAGS_batch_size));

    run(available_tpus.size() < 2 ? available_tpus[0] : available_tpus[1],
                    absl::GetFlag(FLAGS_co_compiled_bird_model_path), 
                    absl::GetFlag(FLAGS_co_compiled_plant_model_path), 
                    absl::GetFlag(FLAGS_bird_image_path), 
                    absl::GetFlag(FLAGS_plant_image_path), 
                    num_inferences, "co-compiled", absl::GetFlag(FLAGS_batch_size));
  }
}
