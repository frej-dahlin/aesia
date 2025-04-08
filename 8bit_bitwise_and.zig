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

    const net = try dlg.Network.initRandom(ally, &[_]usize{ 16, 1024, 512, 256, 128, 64, 32, 16, 8 });
    const beta_1 = 0.9;
    const beta_2 = 0.999;
    const epsilon = 1e-08;
    const adam_m = try ally.alloc([16]f32, net.nodes.len);
    const adam_v = try ally.alloc([16]f32, net.nodes.len);
    for (adam_m, adam_v) |*ml, *vl| {
        @memset(ml, 0);
        @memset(vl, 0);
    }
    const learn_rate = 0.01;
    const batch_size = 32;
    var x: [16]f32 = undefined;
    var y: [8]f32 = undefined;
    while (true) {
        var mse: f32 = 0;
        for (net.nodes) |*node| @memset(&node.gradient, 0);
        for (0..batch_size) |j| {
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
            net.update_gradient(&x, &y);
            for (net.lastLayer(), y) |node, softbit| {
                mse += (node.value - softbit) * (node.value - softbit) / batch_size / 2;
            }
            if (j == batch_size - 1) {
                var softsum: f32 = 0;
                var base: f32 = 1;
                for (net.lastLayer()) |node| {
                    softsum += base * @round(node.value);
                    base *= 2;
                }
                std.debug.print("{d} == {d}\n", .{ sum, softsum });
            }
        }
        for (net.nodes, adam_m, adam_v) |*node, *ml, *vl| {
            inline for (&node.weights, node.gradient, ml, vl) |*w, d, *mw, *vw| {
                mw.* = beta_1 * mw.* + (1 - beta_1) * d / batch_size;
                vw.* = beta_2 * vw.* + (1 - beta_2) * d * d / batch_size / batch_size;
                const m_hat = mw.* / (1 - beta_1);
                const v_hat = vw.* / (1 - beta_2);
                w.* -= learn_rate * m_hat / (@sqrt(v_hat) + epsilon);
            }
        }
        std.debug.print("{d}\n", .{mse});
    }
    net.deinit(ally);
}
