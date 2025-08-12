const std = @import("std");
const assert = std.debug.assert;

const aesia = @import("aesia.zig");

pub fn Convolution(options: struct {
    Layer: type,
    kernel_count: usize,
    dim_in: usize,
}) type {
    const Layer = options.Layer;
    const kernel_count = options.kernel_count;
    const dim_in = options.dim_in;
    const dim_co = dim_in / Layer.info.dim_in;
    const dim_out = kernel_count * dim_co * Layer.info.dim_out;
    comptime {
        assert(Layer.info.trainable);
        // For now we must wrap the layer into a network.
        // This is the most difficult layer type to work with, hence why we are starting
        // with support for it.
        assert(Layer.info.is_network);
    }
    return struct {
        const Self = @This();

        const kernel_parameter_count = Layer.parameter_count;
        pub const info: aesia.layer.Info = .{
            .dim_in = dim_in,
            .dim_out = dim_out,
            .trainable = Layer.info.trainable,
            .parameter_count = if (Layer.info.trainable)
                kernel_count * kernel_parameter_count
            else
                0,
            .parameter_alignment = if (Layer.info.trainable)
                Layer.info.parameter_alignment
            else
                0,
            .statefull = true,
            .in_place = false,
        };

        kernels: [kernel_count]Layer,
        caches: [kernel_count][dim_co]Layer.ForwardCache,

        pub fn init(self: *Self, parameters: *[kernel_count][kernel_parameter_count]f32) void {
            for (&self.kernels, parameters) |*kernel, *parameters_ply|
                kernel.init(parameters_ply);
        }

        pub fn takeParameters(
            self: *Self,
            parameters: *[kernel_count][kernel_parameter_count]f32,
        ) void {
            for (&self.kernels, parameters) |*kernel, *parameters_ply|
                kernel.takeParameters(parameters_ply);
        }

        pub fn giveParameters(
            self: *Self,
        ) void {
            for (&self.kernels) |*kernel|
                kernel.giveParameters();
        }

        pub fn eval(
            self: *Self,
            input: *const [dim_co][Layer.info.dim_in]f32,
            output: *[kernel_count][dim_co][Layer.info.dim_out]f32,
        ) void {
            for (0..kernel_count) |ply| {
                const kernel = &self.kernels[ply];
                for (0..dim_co) |row| {
                    kernel.eval(&input[row], &output[ply][row]);
                }
            }
        }

        pub fn forwardPass(
            self: *Self,
            input: *const [dim_co][Layer.info.dim_in]f32,
            output: *[kernel_count][dim_co][Layer.info.dim_out]f32,
        ) void {
            for (0..kernel_count) |ply| {
                const kernel = &self.kernels[ply];
                const cache = &self.caches[ply];
                for (0..dim_co) |row| {
                    // Note that kernel must be a network layer.
                    kernel.forwardPass(&input[row], &cache[row], &output[ply][row]);
                }
            }
        }

        // Note: There is a possibility of a very dangerous bug here.
        // Both activation_delta and forward_input might be of the same size, then the
        // compiler will not issue a warning in case they are mixed up.
        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [kernel_count][dim_co][Layer.info.dim_out]f32,
            forward_input: *const [dim_co][Layer.info.dim_in]f32,
            gradient: *[kernel_count][kernel_parameter_count]f32,
            argument_delta: *[dim_co][Layer.info.dim_in]f32,
        ) void {
            // FIXME: break API such that we can assume argument_delta is zero-initialized.
            var argument_delta_buffer: [Layer.info.dim_in]f32 = @splat(0);
            argument_delta.* = @splat(@splat(0));
            for (0..kernel_count) |ply| {
                const kernel = &self.kernels[ply];
                const cache = &self.caches[ply];
                for (0..dim_co) |row| {
                    argument_delta_buffer = @splat(0);
                    kernel.backwardPass(
                        &activation_delta[ply][row],
                        &forward_input[ply],
                        &cache[row],
                        &gradient[ply],
                        &argument_delta_buffer,
                    );
                    for (0..Layer.info.dim_in) |j| {
                        argument_delta[row][j] += argument_delta_buffer[j];
                    }
                }
            }
        }

        pub fn backwardPassFinal(
            self: *Self,
            activation_delta: *const [kernel_count][dim_in / Layer.info.dim_in][Layer.info.dim_out]f32,
            gradient: *[kernel_count][kernel_parameter_count]f32,
        ) void {
            for (
                &self.kernels,
                activation_delta,
                gradient,
            ) |*kernel, *activation_delta_ply, *gradient_ply| {
                for (0..dim_in / Layer.info.dim_in) |i| {
                    kernel.backwardPassFinal(&activation_delta_ply[i], gradient_ply);
                }
            }
        }
    };
}
