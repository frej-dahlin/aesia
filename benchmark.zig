const std = @import("std");

const dlg = @import("dlg.zig");

var model: dlg.Model(.{
    .shape = &[_]usize{ 8, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 },
}) = .default;

const iter_count = 10;

pub fn main() !void {
    model.network.randomize(0);
    const x: [8]f32 = .{0} ** 8;

    var timer = try std.time.Timer.start();
    for (0..iter_count) |_| _ = model.eval(&x);
    const eval_ms = timer.read() / 1_000_000;

    const y: [1]f32 = .{0};
    timer.reset();
    for (0..iter_count) |_| model.backprop(&x, &y);
    const update_ms = timer.read() / 1_000_000;

    std.debug.print("1000 Iterations\n", .{});
    std.debug.print("-------------------------\n", .{});
    std.debug.print("eval\t\tbackprop\n", .{});
    std.debug.print("{d}ms\t\t{d}ms\n", .{ eval_ms, update_ms });
}
