const std = @import("std");

const dlg = @import("dlg.zig");

pub fn main() !void {
	const allocator = std.heap.page_allocator;
	var net = try dlg.Network(.{
		//.shape = &[_]usize{8, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1},
		.shape = &[_]usize{8, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1},
		//.shape = &[_]usize{8, 16, 8, 4, 2, 1},
	}).initRandom(allocator);
	const x: [8]f32 = .{0} ** 8;
	const y: [1]f32 = undefined;
	
	var timer = try std.time.Timer.start();
	for (0..1000) |_| net.eval(x);
	const eval_ms = timer.read() / 1_000_000;
	net.nodes.value[0] = 1;
	std.debug.print("{d}\n", .{@sizeOf(@TypeOf(net))});
	
	timer.reset();
	//for (0..1000) |_| net.update_gradient(x, &y);
	_ = y;
	const update_ms = timer.read() / 1_000_000;
	
	std.debug.print("1000 Iterations\n", .{});
	std.debug.print("-------------------------\n", .{});
	std.debug.print("eval\t\tupdate_gradient\n", .{});
	std.debug.print("{d}ms\t\t{d}ms\n", .{eval_ms, update_ms});
}
