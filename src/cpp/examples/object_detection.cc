// An example to detect image.
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/cpp/detection/engine.h"
#include "src/cpp/examples/label_utils.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"
#include <chrono>  // NOLINT

ABSL_FLAG(std::string, model_path,
          "./test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite",
          "Path to the tflite model.");

ABSL_FLAG(std::string, image_path, "./test_data/pets.bmp",
          "Path to the image to be classified.");

ABSL_FLAG(std::string, labels_path, "./test_data/pet_labels.txt",
          "Path to the imagenet labels.");

void ObjectDetection(const std::string& model_path, const std::string& image_path,
                   const std::string& labels_path, const int num_inferences) {
  // Load the model.
  coral::DetectionEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();
  // Read the image.
  std::vector<uint8_t> input_tensor = coral::GetInputFromImage(
      image_path,
      {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});
  // Read the label file.
  auto labels = coral::ReadLabelFile(labels_path);

  // timing 
  const auto& start_time = std::chrono::steady_clock::now();
  for (int i =0; i <= num_inferences; i++) {
    engine.DetectWithInputTensor(input_tensor);
  }
  std::chrono::duration<double> time_span =
      std::chrono::steady_clock::now() - start_time;
  std::cout << "Running " << model_path
	    << " with " << num_inferences << " inferences"
	    << " costs: " << time_span.count() << " seconds." << std::endl;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  for (auto i : {100, 200, 400, 800, 1600, 3200}) {
    ObjectDetection(absl::GetFlag(FLAGS_model_path),
                  absl::GetFlag(FLAGS_image_path),
                  absl::GetFlag(FLAGS_labels_path),
		  i);
  }
}
