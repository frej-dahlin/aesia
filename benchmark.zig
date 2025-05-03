const std = @import("std");

const skiffer = @import("skiffer.zig");
const LogicLayer = skiffer.layer.PackedLogic;
const GroupSum = skiffer.layer.GroupSum;

const Model = skiffer.Model(&.{
    LogicLayer(784, 16_000, .{ .seed = 0 }),
    LogicLayer(16_000, 16_000, .{ .seed = 1 }),
    LogicLayer(16_000, 16_000, .{ .seed = 2 }),
    LogicLayer(16_000, 16_000, .{ .seed = 3 }),
    LogicLayer(16_000, 16_000, .{ .seed = 4 }),
    GroupSum(16_000, 10),
}, .{ .Optimizer = null, .Loss = skiffer.loss.DiscreteCrossEntropy(u8, 10) });

var model: Model = .default;

const count = 1000;

pub fn main() !void {
    model.initParameters();
    _ = count;
    const features: [count][784]f32 = @splat(@splat(0));
    const labels: [count]u8 = @splat(0);
    var timer = try std.time.Timer.start();
    model.train(.init(&features, &labels), .empty, 1, 32);
    std.debug.print("{d}ms\n", .{timer.read() / 1_000_000});
}
