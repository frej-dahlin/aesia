const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const aesia_mod = b.createModule(.{
        .root_source_file = b.path("../src/aesia.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mnist_mod = b.createModule(.{
        .root_source_file = b.path("mnist.zig"),
        .target = target,
        .optimize = optimize,
    });
    mnist_mod.addImport("aesia", aesia_mod);
    const mnist_exe = b.addExecutable(.{
        .name = "mnist",
        .root_module = mnist_mod,
    });
    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const mnist_run_cmd = b.addRunArtifact(mnist_exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    mnist_run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        mnist_run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const mnist_run_step = b.step("mnist", "Run the mnist example");
    mnist_run_step.dependOn(&mnist_run_cmd.step);

    const cifar10_mod = b.createModule(.{
        .root_source_file = b.path("cifar10.zig"),
        .target = target,
        .optimize = optimize,
    });
    cifar10_mod.addImport("aesia", aesia_mod);
    const cifar10_exe = b.addExecutable(.{
        .name = "cifar10",
        .root_module = cifar10_mod,
    });
    const cifar10_run_cmd = b.addRunArtifact(cifar10_exe);
    cifar10_run_cmd.step.dependOn(b.getInstallStep());

    const cifar10_run_step = b.step("cifar10", "Run the cifar10 example");
    cifar10_run_step.dependOn(&cifar10_exe.step);
    if (b.args) |args| {
        cifar10_run_cmd.addArgs(args);
    }
    cifar10_run_step.dependOn(&cifar10_run_cmd.step);
}
