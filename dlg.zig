const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const EnumArray = std.EnumArray;
const MultiArrayList = std.MultiArrayList;
const ArrayList = std.ArrayList;
const AutoArrayHashMap = std.AutoArrayHashMap;

pub const optim = @import("optim.zig");

pub const f32x16 = @Vector(16, f32);

// Note: This is the hottest part of the code, I *think* that AVX-512 should handle it with gusto,
// but testing is required.
pub fn softmax2(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const sigma = @exp2(logit);
    const denom = @reduce(.Add, sigma);
    return sigma / @as(f32x16, @splat(denom));
}

pub fn softmax(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const sigma = @exp(logit);
    const denom = @reduce(.Add, sigma);
    return sigma / @as(f32x16, @splat(denom));
}

// Experimental: fails severly sometimes.
// Computes (1 + x / 256)^256.
pub fn softmax_approx(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const scale = @as(f32x16, @splat(@as(f32, 1.0 / 256.0)));
    var ret = @as(f32x16, @splat(1)) + logit * scale;
    inline for (0..8) |_| ret *= ret;
    return ret;
}

const Gate = struct {
    /// Returns a vector representing all values of a soft logic gate.
    pub fn vector(a: f32, b: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            a * b,
            a - a * b,
            a,
            b - a * b,
            b,
            a + b - 2 * a * b,
            a + b - a * b,
            1 - (a + b - a * b),
            1 - (a + b - 2 * a * b),
            1 - b,
            1 - (b - a * b),
            1 - a,
            1 - (a - a * b),
            1 - a * b,
            1,
        };
    }

    /// Returns the soft gate vector differentiated by the first variable.
    /// Note that it only depends on the second variable.
    pub fn del_a(b: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            b,
            1 - b,
            1,
            -b,
            0,
            1 - 2 * b,
            1 - b,
            -(1 - b),
            -(1 - 2 * b),
            0,
            b,
            -1,
            -(1 - b),
            -b,
            0,
        };
    }

    /// Returns the soft gate vector differentiated by the second variable.
    /// Note that it only depends on the first variable.
    pub fn del_b(a: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            a,
            -a,
            0,
            1 - a,
            1,
            1 - 2 * a,
            1 - a,
            -(1 - a),
            -(1 - 2 * a),
            -1,
            -(1 - a),
            0,
            a,
            -a,
            0,
        };
    }
};

const NetworkOptions = struct {
    shape: []const usize,
    softmax_base_2: bool = true,
};

/// A network of differentiable logic gates.
pub fn Network(comptime options: NetworkOptions) type {
    const shape = options.shape;

    // Assert validity of the shape array.
    if (shape.len < 2) @compileError("options.shape: at least dimensions are required, an input and output");
    for (shape) |dim| if (dim == 0) @compileError("options.shape: all dimensions need to be nonzero");
    for (shape[0 .. shape.len - 1], shape[1..]) |dim, next| {
        if (2 * next < dim) @compileError("options.shape: not allowed to shrink dimensions by factor > 2");
    }

    return struct {
        const Self = @This();

        // Ranges are stored in this manner to make the illegal ranges (where to < from) unrepresentable.
        const Range = struct {
            from: usize,
            len: usize,

            pub fn to(range: Range) usize {
                return range.from + range.len;
            }
        };
        // The following constants are created to deal with the flat struct of arrays architecture
        // Doing it in this manner allows for 0 overhead in memory.
        const layer_count = shape.len - 1;
        // Allows for looping over the network layer by layer in a simple manner.
        const layer_ranges: [layer_count]Range = blk: {
            var ranges: [layer_count]Range = undefined;
            var offset: usize = 0;
            for (&ranges, shape[1..]) |*range, dim| {
                range.* = .{ .from = offset, .len = dim };
                offset += dim;
            }
            break :blk ranges;
        };
        const layer_last = layer_ranges[layer_count - 1];

        // The number of total nodes in the network.
        pub const node_count = layer_last.to();

        // The network will not have a layer with shape[0] number of nodes. But rather the first layer,
        // nodes from 0 .. shape[1] will have parent indices that range from 0 .. shape[0], which is
        // the dimension of the input.
        const dim_in = shape[0];
        // The last layer
        const dim_out = shape[shape.len - 1];
        comptime {
            assert(dim_out == layer_last.len);
        }

        const NodeIndex = std.math.IntFittingRange(0, node_count);
        // We use a struct of arrays for cache efficiency.
        // The parameters that yields the probability of each logic gate.
        logit: [node_count]f32x16 align(64),

        // The cached value of softmax2(logit).
        sigma: [node_count]f32x16 align(64),

        // The gradient of the cost function with respect to the nodes logit.
        gradient: [node_count]f32x16 align(64),

        // The indices of the nodes parents, for the first layer i.e. nodes 0..shape[1] these
        // are instead indices into the input array.
        parents: [node_count][2]NodeIndex,

        // The current feedforward value of the network, computed by eval.
        value: [node_count]f32,

        // The derivative of the cost function with respect to the nodes value.
        // Used for efficient backpropagation of the gradient.
        delta: [node_count]f32,

        pub const default = Self{
            .logit = @splat([_]f32{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10),
            .sigma = @splat(@splat(0)),
            .gradient = @splat(@splat(0)),
            .parents = @splat(.{ 0, 0 }),
            .value = @splat(0),
            .delta = @splat(0),
        };
        /// Updates the gradient of the network given the gradient of
        /// the cost function with respect to the last layer's value.
        /// The caller is responsible correctly setting the delta in
        /// advance.
        pub fn backprop(net: *Self, x: *const [dim_in]f32) void {
            @setFloatMode(.optimized);

            // Compensating factor for softmax if using base 2.
            const tau = if (options.softmax_base_2) @log(2.0) else 1.0;

            @memset(net.delta[0..layer_last.from], 0);
            // Propagate the delta backwards, layer by layer, and use it to compute the gradient.
            // Inlining this loop ensures that we avoid the inner branch that is only valid for
            // the first layer.
            comptime var l: usize = layer_count;
            inline while (l > 0) {
                l -= 1;
                const range = layer_ranges[l];
                const input = if (l == 0) x else net.value;
                for (range.from..range.to()) |j| {
                    const parents = net.parents[j];
                    const a = input[parents[0]];
                    const b = input[parents[1]];
                    const sigma = net.sigma[j];
                    const delta = net.delta[j];
                    // Children are responsible for updating their parents delta.
                    // In this way we avoid storing a list of all children for each node.
                    if (l != 0) {
                        net.delta[parents[0]] += delta * @reduce(.Add, sigma * Gate.del_a(b));
                        net.delta[parents[1]] += delta * @reduce(.Add, sigma * Gate.del_b(a));
                    }
                    const value = net.value[j];
                    const gate = Gate.vector(a, b);
                    net.gradient[j] += @as(f32x16, @splat(tau * delta)) * sigma * (gate - @as(f32x16, @splat(value)));
                }
            }
        }

        /// Evaluates the network, caching sigma and the value of each node.
        pub fn eval(net: *Self, x: *const [dim_in]f32) *const [dim_out]f32 {
            @setFloatMode(.optimized);
            // This inline for loop avoids two separate for loops with the same body
            // except input being different for the first layer, nodes from 0 to shape[i].
            // Note: One single for-loop is faster, however, this structure is
            // in preparation for multithreading.
            inline for (layer_ranges, 0..) |range, l| {
                const input = if (l == 0) x else net.value;
                for (range.from..range.to()) |j| {
                    const parents = net.parents[j];
                    const a = input[parents[0]];
                    const b = input[parents[1]];
                    const logit = net.logit[j];
                    net.sigma[j] = if (options.softmax_base_2) softmax2(logit) else softmax(logit);
                    net.value[j] = @reduce(.Add, net.sigma[j] * Gate.vector(a, b));
                }
            }
            return @ptrCast(net.value[layer_last.from..layer_last.to()]);
        }

        /// Randomizes the connections between each layer in a uniform manner.
        /// Each node of the same layer has equal +- number of children.
        /// It is possible that a node's parents are equal.
        pub fn randomize(net: *Self, seed: u64) void {
            var sfc64 = std.Random.Sfc64.init(seed);
            const rand = sfc64.random();
            net.sequentialize();
            inline for (layer_ranges) |range| {
                rand.shuffle(NodeIndex, @as(*[2 * range.len]NodeIndex, @ptrCast(net.parents[range.from..range.to()])));
            }
        }

        // Sets the connections between each layer to be sequential (wrapping).
        pub fn sequentialize(net: *Self) void {
            inline for (layer_ranges, shape[0 .. shape.len - 1], 0..) |range, dim_prev, l| {
                const offset = if (l == 0) 0 else range.from - dim_prev;
                var i: usize = 0;
                for (net.parents[range.from..range.to()]) |*parents| {
                    const first = i % dim_prev + offset;
                    const second = (i + 1) % dim_prev + offset;
                    parents.* = .{ @truncate(first), @truncate(second) };
                    i += 2;
                }
            }
        }
    };
}

pub fn RelaxBits(bit_count: usize) type {
    return struct {
        const Self = @This();
        const dim = bit_count;
        const Type = [dim]bool;

        // Fixme: use SIMD.
        pub fn eval(self: *Self, x: *const Type) *const [dim]f32 {
            for (self.value, x) |*softbit, bit| softbit.* = if (bit) 1 else 0;
            return &self.value;
        }
    };
}

pub fn InputIdentity(dim_out: usize) type {
    return struct {
        const Self = @This();
        const dim = dim_out;
        const Type = [dim]f32;

        pub fn eval(_: *Self, x: *const Type) *const Type {
            return x;
        }
    };
}

pub fn HalvedMeanSquareError(dim: usize) type {
    return struct {
        // Fixme: Use SIMD.
        pub fn eval(y_pred: *const [dim]f32, y_real: *const [dim]f32) f32 {
            assert(y_pred.len == y_real.len);
            var result: f32 = 0;
            for (y_pred, y_real) |pred, real| result += (pred - real) * (pred - real);
            return result / 2;
        }

        pub fn gradient(y_pred: *const [dim]f32, y_real: *const [dim]f32, result: *[dim]f32) void {
            for (y_pred, y_real, result) |pred, real, *r| r.* = pred - real;
        }
    };
}

pub fn OutputIdentity(dim_in: usize) type {
    return struct {
        const Self = @This();
        const dim = dim_in;
        const Type = [dim]f32;

        pub fn eval(_: *Self, z: *const Type) *const Type {
            return z;
        }
    };
}

pub fn GroupSum(dim_out: usize) type {
    if (dim_out == 0) @compileError("GroupSum output dimension needs to be nonzero");
    return struct {
        pub fn Init(dim_in: usize) type {
            if (dim_in % dim_out != 0) @compileError("GroupSum input dimension must be evenly divisible by output dimension");
            return struct {
                const Self = @This();
                value: [dim_out]f32,
                delta: [dim_out]f32,

                pub fn eval(self: *Self, in: *const [dim_in]f32, out: *const [dim_out]f32) void {
                    const quot = dim_in / dim_out;
                    const denom: f32 = @floatFromInt(dim_in / dim_out);
                    @memset(&self.value, 0);
                    inline for (0..dim_out) |k| {
                        for (k * quot..(k + 1) * quot) |i| out[k] += in[i];
                        out[k] /= denom;
                    }
                    return &self.value;
                }
            };
        }
    }.Init;
}

// Fixme: Add FloatType as an option.
pub const ModelOptions = struct {
    shape: []const usize,
    InputLayer: ?fn (usize) type = null,
    OutputLayer: ?fn (usize) type = null,
    Optimizer: ?fn (type) type = null,
    Cost: ?type = null,
};

pub fn Model(comptime options: ModelOptions) type {
    return struct {
        const Self = @This();
        // Unpack options.
        const shape = options.shape;
        pub const NetworkType = Network(.{ .shape = shape });
        const InputLayer = if (options.InputLayer) |IL| IL(NetworkType.dim_in) else InputIdentity(NetworkType.dim_in);
        const OutputLayer = if (options.OutputLayer) |OL| OL(NetworkType.dim_out) else OutputIdentity(NetworkType.dim_out);
        const OptimizerType = if (options.Optimizer) |O| O(Self) else optim.GradientDescent(.default)(Self);

        const InputType = InputLayer.Type;
        const OutputType = OutputLayer.Type;
        const Cost = if (options.Cost) |C| C else HalvedMeanSquareError(OutputLayer.dim);

        pub const Dataset = struct {
            input: []InputType,
            output: []OutputType,

            pub const empty = Dataset{ .input = &.{}, .output = &.{} };
        };

        network: Network(.{
            .shape = shape,
        }),
        layer_input: InputLayer,
        layer_output: OutputLayer,
        dataset_training: Dataset,
        dataset_validate: Dataset,
        optimizer: OptimizerType,

        pub const default = Self{
            .network = .default,
            .layer_input = undefined,
            .layer_output = undefined,
            .dataset_training = .empty,
            .dataset_validate = .empty,
            .optimizer = .default,
        };

        /// Computes the gradient of the cost function with respect to the network's node's logit.
        /// The result is added to model.network.gradient, the caller is responsible for zeroing
        /// the gradient if desired. In this manner batching of multiple datapoints is possible.
        pub fn backprop(model: *Self, x: *const InputType, y: *const OutputType) void {
            const net = &model.network;
            const last = NetworkType.layer_last;
            // We branch here to avoid forcing OutputIdentity to include useless data
            // as well as memcpy it to the network.
            if (options.OutputLayer == null) {
                Cost.gradient(model.eval(x), y, @ptrCast(net.delta[last.from..last.to()]));
            } else {
                Cost.gradient(model.eval(x), y, &net.layer_output.delta);
                model.layer_output.backprop(net.values[last.from..last.to()], net.delta[last.from..last.to()]);
            }
            model.network.backprop(x);
        }

        /// The cost of the model given some input and its expected output.
        pub fn cost(model: *Self, x: *const InputType, y_real: *const OutputType) f32 {
            return Cost.eval(model.eval(x), y_real);
        }

        /// Evaluates the model.
        pub fn eval(model: *Self, x: *const InputType) *const OutputType {
            const a = model.layer_input.eval(x);
            const b = model.network.eval(a);
            return model.layer_output.eval(b);
        }

        pub fn train(model: *Self, epoch_count: usize) void {
            const validate = model.dataset_validate;
            assert(validate.input.len == validate.output.len);
            const scale: f32 = 1.0 / @as(f32, @floatFromInt(validate.input.len));
            for (0..epoch_count) |epoch| {
                model.optimizer.step(model);
                var loss: f32 = 0;
                for (validate.input, validate.output) |x, y_real| {
                    const y_pred = model.eval(&x);
                    loss += Cost.eval(y_pred, &y_real);
                }
                std.debug.print("epoch: {d}\tloss: {d}\n", .{ epoch, loss * scale });
            }
        }
    };
}
