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

    const net = try dlg.Network.initRandom(ally, &[_]usize{ 16, 512, 256, 128, 64, 32, 16, 8 });
    for (net.layers) |layer| {
    	@memset(layer.items(.adam_m), [_]f32{0} ** 16);
    	@memset(layer.items(.adam_v), [_]f32{0} ** 16);
    }
    const beta_1 = 0.9;
    const beta_2 = 0.999;
    const epsilon = 1e-08;
    const learn_rate = 0.01;
    const batch_size = 1;
    var x: [16]f32 = undefined;
    var y: [8]f32 = undefined;
    var niter: usize = 0;
    while (true) {
        niter += 1;
        var mse: f32 = 0;
        for (net.layers) |layer| @memset(layer.items(.gradient), [_]f32{0} ** 16);
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
            net.update_gradient(&x, &y);
            for (net.lastLayer().items(.value), y) |value, softbit| {
                mse += (value - softbit) * (value - softbit) / batch_size / 2;
            }
            //if (j == batch_size - 1) {
            //var softsum: f32 = 0;
            //var base: f32 = 1;
            //for (net.lastLayer()) |node| {
            //softsum += base * @round(node.value);
            //base *= 2;
            //}
            //std.debug.print("{d} == {d}\n", .{ sum, softsum });
            //}
        }
        for (net.layers) |layer| {
            for (layer.items(.weights), layer.items(.gradient), layer.items(.adam_m), layer.items(.adam_v)) |*weights, gradient, *m, *v| {
                const denom: f32 = 1;
                m.* = @as(dlg.v16f32, @splat(beta_1)) * m.* +
                    @as(dlg.v16f32, @splat((1 - beta_1) / denom)) * gradient;
                v.* = @as(dlg.v16f32, @splat(beta_2)) * v.* +
                    @as(dlg.v16f32, @splat((1 - beta_2) / denom / denom)) * gradient * gradient;
                const m_hat = m.* / @as(dlg.v16f32, @splat(1 - beta_1));
                const v_hat = v.* / @as(dlg.v16f32, @splat(1 - beta_2));
                weights.* -= @as(dlg.v16f32, @splat(learn_rate)) *
	                m_hat / (@sqrt(v_hat) + @as(dlg.v16f32, @splat(epsilon)));
            }
        }
        if (niter % 1000 == 0) std.debug.print("{d}\n", .{mse});
    }
    net.deinit(ally);
}
