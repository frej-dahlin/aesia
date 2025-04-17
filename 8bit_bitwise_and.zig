const std = @import("std");

const dlg = @import("dlg.zig");

var model: dlg.Model(.{
    .shape = &(.{16} ++ .{128} ** 8 ++ .{ 64, 32, 16, 8 }),
    .Optimizer = dlg.optim.Adam(.{ .learn_rate = 0.01 }),
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
    const set_count = 10000;
    const training_count = 9000;
    var dataset: [set_count]@TypeOf(model).Datapoint = undefined;
    for (&dataset) |*point| {
        const x = &point.input;
        const y = &point.output;
        const a = rand.int(u8);
        const b = rand.int(u8);
        const c = a +% b;
        var base: usize = 1;
        for (x[0..8], x[8..], y) |*first, *second, *out| {
            first.* = if (a & base != 0) 1 else 0;
            second.* = if (b & base != 0) 1 else 0;
            out.* = if (c & base != 0) 1 else 0;
            base *= 2;
        }
    }
    model.train(dataset[0..training_count], dataset[training_count..], 1000, 32);

    for (dataset[training_count..]) |point| {
        for (model.eval(&point.input)) |softbit| std.debug.print("{d}", .{@round(softbit)});
        std.debug.print(" == ", .{});
        for (point.output) |softbit| std.debug.print("{d}", .{@round(softbit)});
        std.debug.print("\n", .{});
    }
}
