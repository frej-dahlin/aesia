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

    const net = try dlg.Network.initRandom(ally, &[_]usize{ 16, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 16, 8});
    const beta_1 = 0.9;
    const beta_2 = 0.999;
    const epsilon = 1e-08;
    const adam_m = try ally.alloc(dlg.v16f32, net.nodes.len);
    const adam_v = try ally.alloc(dlg.v16f32, net.nodes.len);
    for (adam_m, adam_v) |*ml, *vl| {
        ml.* = [_]f32{0} ** 16;
        vl.* = [_]f32{0} ** 16;
    }
    const learn_rate = 0.01;
    const batch_size = 1;
    var x: [16]f32 = undefined;
    var y: [8]f32 = undefined;
    var niter: usize = 0;
    while (true) {
    	niter += 1;
        var mse: f32 = 0;
        @memset(net.nodes.items(.gradient), [_]f32{0} ** 16);
        for (0..batch_size) |j| {
        	_ = j;
            const a: u8 = @truncate(rand.uintLessThan(usize, std.math.maxInt(u8)));
            const b: u8 = @truncate(rand.uintLessThan(usize, std.math.maxInt(u8)));
            const sum: u8 = a +% b;
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
            for (net.nodes.items(.value)[net.nodes.len - net.shape[net.shape.len - 1]..], y) |value, softbit| {
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
        for (net.nodes.items(.weights), net.nodes.items(.gradient), adam_m, adam_v) |*weights, gradient, *ml, *vl| {
            const denom: f32 = 1; //batch_size;
            ml.* = @as(dlg.v16f32, @splat(beta_1)) * ml.* +
                @as(dlg.v16f32, @splat((1 - beta_1) / denom)) *
                    gradient;
            vl.* = @as(dlg.v16f32, @splat(beta_2)) * vl.* +
                @as(dlg.v16f32, @splat((1 - beta_2) / denom / denom)) *
                    gradient * gradient;
            const m_hat = ml.* / @as(dlg.v16f32, @splat(1 - beta_1));
            const v_hat = vl.* / @as(dlg.v16f32, @splat(1 - beta_2));
            weights.* -= @as(dlg.v16f32, @splat(learn_rate)) * m_hat /
                (@sqrt(v_hat) + @as(dlg.v16f32, @splat(epsilon)));
        }
        if (niter % 1000 == 0) std.debug.print("{d}\n", .{mse});
    }
    net.deinit(ally);
}
