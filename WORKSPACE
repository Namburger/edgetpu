workspace(name = "edgetpu")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

TENSORFLOW_COMMIT = "d855adfc5a0195788bf5f92c3c7352e638aa1109"
# Command to calculate: curl -OL <FILE-URL> | sha256sum | awk '{print $1}'
TENSORFLOW_SHA256 = "b8a691dbea2bb028fa8f7ce407b70ad236dae0a8705c8010dc7bad8af7e93bac"

# Be consistent with tensorflow/WORKSPACE.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

# Be consistent with tensorflow/WORKSPACE.
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases


http_archive(
    name = "org_tensorflow",
    sha256 = TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-" + TENSORFLOW_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/" + TENSORFLOW_COMMIT + ".tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

new_local_repository(
    name = "libedgetpu",
    path = "libedgetpu",
    build_file = "libedgetpu/BUILD"
)

local_repository(
    name = "edgetpu_swig",
    path = "edgetpu/swig",
)

new_local_repository(
    name = "python_linux",
    path = "/usr/include",
    build_file = "BUILD.python"
)

local_repository(
    name = "tools",
    path = "tools",
)
load("@tools//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool")

