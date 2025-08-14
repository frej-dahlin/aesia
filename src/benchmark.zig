const std = @import("std");

const aesia = @import("aesia.zig");
const LogicLayer = aesia.layer.Logic;
const PackedLogicLayer = aesia.layer.PackedLogic;
const GroupSum = aesia.layer.GroupSum;

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();

const ModelOld = aesia.Model(&.{
    LogicLayer(784, 16_000, .{ .rand = &rand }),
    LogicLayer(16_000, 16_000, .{ .rand = &rand }),
    LogicLayer(16_000, 16_000, .{ .rand = &rand }),
    LogicLayer(16_000, 16_000, .{ .rand = &rand }),
    LogicLayer(16_000, 16_000, .{ .rand = &rand }),
    GroupSum(16_000, 10),
}, .{ .Optimizer = null, .Loss = aesia.loss.DiscreteCrossEntropy(u8, 10) });

const ModelNew = aesia.Model(
    &.{
        PackedLogicLayer(784, 16_000, .{ .rand = &rand }),
        PackedLogicLayer(16_000, 16_000, .{ .rand = &rand }),
        PackedLogicLayer(16_000, 16_000, .{ .rand = &rand }),
        PackedLogicLayer(16_000, 16_000, .{ .rand = &rand }),
        PackedLogicLayer(16_000, 16_000, .{ .rand = &rand }),
        GroupSum(16_000, 10),
    },
    .{
        .Optimizer = null,
        .Loss = aesia.loss.DiscreteCrossEntropy(u8, 10),
    },
);

var model_old: ModelOld = undefined;
var model_new: ModelNew = undefined;

const count = 1000;

pub fn main() !void {
    model_old.init();
    model_new.init();
    _ = count;
    const features: [count][784]f32 = @splat(@splat(0));
    const labels: [count]u8 = @splat(0);

    var timer = try std.time.Timer.start();
    model_old.train(.init(&features, &labels), .empty, 1, 32);
    std.debug.print("legacy gates: \t{d}ms\n", .{timer.read() / 1_000_000});

    timer.reset();
    model_new.train(.init(&features, &labels), .empty, 1, 32);
    std.debug.print("packed gates: \t{d}ms\n", .{timer.read() / 1_000_000});
}
