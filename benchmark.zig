const std = @import("std");

const dlg = @import("dlg.zig");

var model: dlg.Model(.{
    .shape = &[_]usize{ 8, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 },
}) = .default;

pub fn main() !void {
    model.network.randomize(0);
    const dataset: [1000]@TypeOf(model).Datapoint = undefined;

    var timer = try std.time.Timer.start();
    model.train(&dataset, &.{}, 1, 100);
    const differentiate_ms = timer.read() / 1_000_000;

    std.debug.print("1000 Iterations\n", .{});
    std.debug.print("-------------------------\n", .{});
    std.debug.print("differentiate\n", .{});
    std.debug.print("{d}ms\n", .{differentiate_ms});
}
