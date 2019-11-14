// An example to detect image.
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "src/cpp/detection/engine.h"
#include "src/cpp/examples/label_utils.h"
#include "src/cpp/examples/model_utils.h"
#include "src/cpp/test_utils.h"

#include <dirent.h> // a little ugly but will do for now
#include <stdio.h>
#include <vector>

ABSL_FLAG(std::string, model_path,
          "./test_data/ssd_mobilenet_v1_fine_tuned_edgetpu.tflite",
          "Path to the tflite model.");

ABSL_FLAG(std::string, images_path, "./test_data/bmp/",
          "Path to the image to be classified.");

// this is a little ugly
const std::vector<std::string> get_images(const std::string& images_path) {
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;

  std::vector<std::string> ret;

  dp = opendir(images_path.c_str());
  if (dp != nullptr) {
    while ((entry = readdir(dp))) {
      if (entry->d_name[0] != '.') // so hacky :(
        ret.push_back({images_path + entry->d_name});
    }
  }

  return ret;
}

void ObjectDetection(const std::string& model_path, const std::string& images_path) {
  // Load the model.
  coral::DetectionEngine engine(model_path);
  std::vector<int> input_tensor_shape = engine.get_input_tensor_shape();

  // Get all images
  const auto& images = get_images(images_path);
  for (auto& image_path : images) {
    std::cout << "Image: " << image_path << "\n";
    // read current images
    std::vector<uint8_t> input_tensor = coral::GetInputFromImage(
        image_path,
        {input_tensor_shape[1], input_tensor_shape[2], input_tensor_shape[3]});

    auto results = engine.DetectWithInputTensor(input_tensor);
    for (const auto& result : results) {
      std::cout << "Score: " << result.score << std::endl;
      std::cout << "Corner: " << result.corners.DebugString() << std::endl;
    }
    std::cout << "---------------------------" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  ObjectDetection(absl::GetFlag(FLAGS_model_path),
                absl::GetFlag(FLAGS_images_path));
}
