const std = @import("std");

const dlg = @import("dlg.zig");

pub fn main() !void {
    const ally = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    var net = try dlg.Network(.{
    	.shape = &[_]usize{ 16, 512, 256, 128, 64, 32, 16, 8},
    }).initRandom(ally);
    const learn_rate = 0.01;
    const batch_size = 1;
    var x: [16]f32 = undefined;
    var y: [8]f32 = undefined;
    var niter: usize = 0;
    while (true) {
        niter += 1;
        var mse: f32 = 0;
        @memset(&net.nodes.gradient, [_]f32{0} ** 16);
        for (0..batch_size) |j| {
            _ = j;
            const a: u8 = @truncate(rand.uintLessThan(usize, std.math.maxInt(u8)));
            const b: u8 = @truncate(rand.uintLessThan(usize, std.math.maxInt(u8)));
            const sum: u8 = a & b;
            inline for (x[0..8], 0..) |*softbit, i| {
                softbit.* = if ((1 << i) & a != 0) 1 else 0;
            }
            inline for (x[8..], 0..) |*softbit, i| {
                softbit.* = if ((1 << i) & b != 0) 1 else 0;
            }
            inline for (&y, 0..) |*softbit, i| {
                softbit.* = if ((1 << i) & sum != 0) 1 else 0;
            }
            net.update_gradient(x, &y);
            mse += net.cost(&y);
        }
        for (&net.nodes.logits, net.nodes.gradient) |*logits, gradient| logits.* -= @as(dlg.f32x16, @splat(learn_rate)) * gradient;
        if (niter % (1000 / batch_size) == 0) std.debug.print("{d}\n", .{mse});
    }
    net.deinit(ally);
}
