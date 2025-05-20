const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const aesia = @import("aesia.zig");

const dim = 32;
const State = [32][32]f32;

const data_count = 1_000;
const training_count = 1_000;

var features: [data_count]State = undefined;
var labels: [data_count]State = undefined;

const ConvolutionLogic = aesia.layer.ConvolutionLogic;
const MultiLogicGate = aesia.layer.MultiLogicGate;
const MultiLogicMax = aesia.layer.MultiLogicMax;
const LogicLayer = aesia.layer.PackedLogic;
const GroupSum = aesia.layer.GroupSum;
const MaxPool = aesia.layer.MaxPool;

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();
const width = 2000;
const Model = aesia.Model(&.{
    ConvolutionLogic(32, 32),
}, .{
    .Loss = aesia.loss.HalvedMeanSquareError(32 * 32),
    .Optimizer = aesia.optimizer.Adam(.{ .learn_rate = 0.05 }),
});
var model: Model = undefined;

var padded: [dim + 2][dim + 2]f32 = undefined;

pub fn gol(state: *const State, next: *State) void {
    const Point = @Vector(2, usize);
    const offsets = [9]Point{
        .{ 0, 0 },
        .{ 0, 1 },
        .{ 0, 2 },
        .{ 1, 0 },
        .{ 1, 1 },
        .{ 1, 2 },
        .{ 2, 0 },
        .{ 2, 1 },
        .{ 2, 2 },
    };

    padded = @splat(@splat(0));
    for (padded[1 .. padded.len - 1], state) |*padded_row, row| {
        @memcpy(padded_row[1 .. padded_row.len - 1], &row);
    }
    for (0..dim) |row| {
        for (0..dim) |col| {
            var count: usize = 0;
            const center = Point{ row, col };
            inline for (offsets) |offset| {
                const point = center + offset;
                count += @intFromBool(padded[point[0]][point[1]] == 1);
            }
            if (state[row][col] == 0) {
                next[row][col] = if (count == 3) 1 else 0;
            } else {
                next[row][col] = if (count == 2 or count == 3) 1 else 0;
            }
        }
    }
}

pub fn main() !void {
    model.init();

    for (&features) |*feature| {
        for (0..dim) |row| {
            for (0..dim) |col| {
                feature[row][col] = if (rand.boolean()) 1 else 0;
            }
        }
    }
    for (features, &labels) |feature, *label| gol(&feature, label);

    var timer = try std.time.Timer.start();
    const epoch_count = 100;
    const batch_size = 1;
    model.train(
        .init(@ptrCast(features[0..training_count]), @ptrCast(labels[0..training_count])),
        .init(@ptrCast(features[0..training_count]), @ptrCast(labels[0..training_count])),
        epoch_count,
        batch_size,
    );

    std.debug.print("Training took: {d}min\n", .{timer.read() / std.time.ns_per_min});

    std.debug.print("Writing model to mnist.model\n", .{});
    try model.writeToFile("mnist.model");
}
