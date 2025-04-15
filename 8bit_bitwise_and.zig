const std = @import("std");

const dlg = @import("dlg.zig");

var model: dlg.Model(.{
    .shape = &[_]usize{ 16, 256, 128, 64, 32, 16, 8 },
    .Optimizer = dlg.optim.GradientDescent(.{ .learn_rate = 0.01 }),
}) = .default;

pub fn main() !void {
    model.network.randomize(0);
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    // Generate datasets.
    const set_count = 10000;
    const training_count = 9000;
    var xs: [set_count][16]f32 = undefined;
    var ys: [set_count][8]f32 = undefined;
    for (&xs, &ys) |*x, *y| {
        const a = rand.int(u8);
        const b = rand.int(u8);
        const c = a & b;
        inline for (x[0..8], x[8..], y, 0..) |*first, *second, *out, i| {
            first.* = if ((1 << i) & a != 0) 1 else 0;
            second.* = if ((1 << i) & b != 0) 1 else 0;
            out.* = if ((1 << i) & c != 0) 1 else 0;
        }
    }
    model.dataset_training = .{ .input = xs[0..training_count], .output = ys[0..training_count] };
    model.dataset_validate = .{ .input = xs[training_count..], .output = ys[training_count..] };

    model.train(1000);
}
