const std = @import("std");

const dlg = @import("dlg.zig");

var model: dlg.Model(.{
    .shape = &(.{16} ++ .{512} ** 8),
    .Optimizer = dlg.optim.Adam(.{ .learn_rate = 0.02 }),
    .OutputLayer = dlg.output_layer.GroupSum(0.25, 2),
    .Loss = dlg.loss_function.DiscreteCrossEntropy(u8, 2),
}) = .default;

pub fn main() !void {
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();
    model.network.randomize(rand.int(u64));

    // Generate dataset.
    const set_count = 20000;
    const training_count = 18000;
    var features: [set_count][16]f32 = undefined;
    var labels: [set_count]u8 = undefined;
    for (&features, &labels) |*feature, *label| {
        const x = rand.int(u16);
        var base: usize = 1;
        for (feature) |*softbit| {
            softbit.* = if (x & base != 0) 1 else 0;
            base *= 2;
        }
        label.* = @intFromBool(x % 17 == 0);
    }
    model.train(.init(features[0..training_count], labels[0..training_count]), .init(features[training_count..], labels[training_count..]), 1000, 32);
}
