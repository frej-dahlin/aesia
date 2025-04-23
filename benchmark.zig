const std = @import("std");

const dlg = @import("dlg.zig");
const Logic = dlg.Logic;

// zig fmt: off
// 
const Model = dlg.Model(&.{
      Logic(.{ .input_dim = 784,    .output_dim = 16_000, .seed = 0 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 1 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 2 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 3 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 4 }),
      Logic(.{ .input_dim = 16_000, .output_dim = 10,     .seed = 5 }),
}, .default);
// zig fmt: on

var model: Model = undefined;

const count = 1000;

pub fn main() !void {
    const features: [count][784]f32 = undefined;
    const labels: [count][10]f32 = undefined;
    var timer = try std.time.Timer.start();
    model.train(.init(&features, &labels), .init(&features, &labels), 1, 32);
    std.debug.print("{d}ms\n", .{timer.read() / std.time.ns_per_ms});
}
