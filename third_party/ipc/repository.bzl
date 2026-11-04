"""Pinned CPU IPC Toolkit source bundle; all downloads occur at repository fetch."""

def _ipc_sources_impl(ctx):
    for src in json.decode(ctx.read(ctx.attr.manifest)):
        ctx.download_and_extract(
            url = src["url"],
            sha256 = src["sha256"],
            stripPrefix = src["strip_prefix"],
            output = src["name"],
            type = "tar.gz",
        )
    # Upstream Bazel packages would hide their source trees from the root glob.
    ctx.delete("onetbb/BUILD.bazel")
    for patch in ctx.attr.patches:
        ctx.patch(patch, strip = 1)
    # Share exactly the Eigen headers used by the rest of the repository.
    eigen_root = ctx.path(ctx.attr.eigen_header).dirname.dirname
    ctx.symlink(eigen_root.get_child("Eigen"), "eigen/Eigen")
    ctx.symlink(eigen_root.get_child("unsupported"), "eigen/unsupported")
    ctx.file("CMakeLists.txt", ctx.read(ctx.attr.cmake))
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

ipc_sources = repository_rule(
    implementation = _ipc_sources_impl,
    attrs = {
        "manifest": attr.label(default = Label("//third_party/ipc:sources.json")),
        "cmake": attr.label(default = Label("//third_party/ipc:CMakeLists.txt")),
        "build_file": attr.label(default = Label("//third_party/ipc:toolkit.BUILD.bazel")),
        "eigen_header": attr.label(default = Label("@eigen//:Eigen/Core")),
        "patches": attr.label_list(default = [Label("//third_party/ipc:edge_edge_canonicalization.patch")]),
    },
)
