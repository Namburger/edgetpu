cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "label_utils",
    srcs = ["label_utils.cc"],
    hdrs = ["label_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)
cc_binary(
    name = "minimal",
    testonly = 1,
    srcs = ["minimal.cc"],
    deps = [
        ":model_utils",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)
