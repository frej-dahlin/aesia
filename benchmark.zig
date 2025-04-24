const std = @import("std");

const dlg = @import("dlg.zig");
const Logic = dlg.Logic;
const GroupSum = dlg.GroupSum;

// zig fmt: off
const Model = dlg.Model(&.{
      Logic(.{ .input_dim = 784,    .output_dim = 16_000, .seed = 0 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 1 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 2 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 3 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 4 }),
      GroupSum(16_000, 10),
}, .{.Loss = dlg.loss_function.DiscreteCrossEntropy(u8, 10), .Optimizer = null});
// zig fmt: on

var model: Model = undefined;

const count = 1000;

pub fn main() !void {
    model.init();
    _ = count;
    const features: [count][784]f32 = undefined;
    const labels: [count]u8 = undefined;
    var timer = try std.time.Timer.start();
    model.train(.init(&features, &labels), .empty, 1, 32);
    std.debug.print("{d}ms\n", .{timer.read() / std.time.ns_per_ms});
}
