const std = @import("std");

const dlg = @import("dlg.zig");
const LogicGates = dlg.LogicGates;

// zig fmt: off
const Network = dlg.Network(.{
     .Layers = &.{
          LogicGates(.{ .input_dim = 784, .output_dim = 16_000, .seed = 0 }),
          LogicGates(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 1 }),
          LogicGates(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 2 }),
          LogicGates(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 3 }),
          LogicGates(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 4 }),
          LogicGates(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 5 }),
          LogicGates(.{ .input_dim = 16_000, .output_dim = 10, .seed = 6 }),
    }
});
// zig fmt: on

var network: Network = undefined;

pub fn main() !void {
    var parameters: [Network.parameter_count]f32 align(64) = @splat(0);
    const input: [784]f32 = @splat(0);
    network.takeParameters(&parameters);
    for (0..1000) |_| _ = network.eval(&input);
}
