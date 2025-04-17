const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const EnumArray = std.EnumArray;
const MultiArrayList = std.MultiArrayList;
const ArrayList = std.ArrayList;
const AutoArrayHashMap = std.AutoArrayHashMap;

pub const input_layer = @import("input.zig");
pub const output_layer = @import("output.zig");
pub const loss_function = @import("loss.zig");
pub const optim = @import("optim.zig");

pub const f32x16 = @Vector(16, f32);

/// Ranges are stored in this manner to make the illegal ranges (where to < from) unrepresentable.
const Range = struct {
    from: usize,
    len: usize,

    pub fn to(range: Range) usize {
        return range.from + range.len;
    }
};

pub fn maxNormalize(v: f32x16) f32x16 {
    return v - @as(f32x16, @splat(@reduce(.Max, v)));
}

// Note: This is the hottest part of the code, I *think* that AVX-512 should handle it with gusto,
// but testing is required.
// Future note: No it is not! Computing the probabilities is amortized over every batch.
// So for decent size batches (say 32+) it is pretty much negligble.
pub fn softmax2(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const sigma = @exp2(maxNormalize(logit));
    const denom = @reduce(.Add, sigma);
    return sigma / @as(f32x16, @splat(denom));
}

pub fn softmax(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const sigma = @exp(maxNormalize(logit));
    const denom = @reduce(.Add, sigma);
    return sigma / @as(f32x16, @splat(denom));
}

// Computes (1 + x / 256)^256.
pub fn softmax_approx(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const scale = @as(f32x16, @splat(@as(f32, 1.0 / 256.0)));
    var ret = @as(f32x16, @splat(1)) + maxNormalize(logit) * scale;
    inline for (0..8) |_| ret *= ret;
    return ret / @as(f32x16, @splat(@reduce(.Add, ret)));
}

const Gate = struct {
    const count = 16;
    /// Returns a vector representing all values of a soft logic gate.
    // Fixme: More optimal construction.
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
    // Fixme: More optimal construction.
    pub fn diff_a(b: f32) f32x16 {
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
    // Fixme: More optimal construction.
    pub fn diff_b(a: f32) f32x16 {
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
    softmax_base_2: bool = false,
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
        // Each node is in a superposition of 16 possible logic gates;
        pub const parameter_count = Gate.count * node_count;
        // Default logits that result in a bias to the passthrough gates.
        // This stabilizes training of the network.
        pub const passtrough_biased: [node_count]f32x16 = @splat(.{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10);

        // The network does not have a layer with shape[0] number of nodes. But rather the first layer,
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

        // The probability distribution of each logic gate.
        // These are derived from a vector of 16 parameters called a logit vector,
        // which are *not* stored, using the formula sigma = softmax(logit).
        // We do it in this way for two reasons:
        // 1. Computing softmax is expensive, however, its cost is amortized over the number
        //    of evaluations between every update. So training the network with
        //    a decent batch size (say 16+) ensures that the preprocessing required
        //    is negligble.
        // 2. Decoupling the logits from the network allows for it to fit into a
        //    more complex model more easily. The model can own all parameters,
        //    including logits and potentially others, in a single flat array.
        sigma: [node_count]f32x16 align(64),

        // The gradient of the node's value with respect to its logit vector.
        gradient: [node_count]f32x16 align(64),

        // The derivative of the node's value with respect to each respective input.
        diff_a: [node_count]f32,
        diff_b: [node_count]f32,

        // The derivative of the loss function with respect to the node's value.
        // The last layer's delta needs to be computed by a model containing the network.
        // The remaining layer's delta are computed by backwardPass.
        delta: [node_count]f32,

        // The indices of the node's parents, for the first layer i.e. nodes 0..shape[1] these
        // are instead indices into the input.
        parents: [node_count][2]NodeIndex,

        // The value of each node, cached for efficiency, used in eval and forwardPass.
        // Note: It is possible, but inconvenient to only store the values in an array
        // whose length is the maximum node_count layerwise.
        value: [node_count]f32,

        // A default network with uniform probabilities for each gate.
        // Before using the network on must call randomizeConnections to
        // initialize every node's parents.
        pub const default = Self{
            .sigma = @splat(@splat(1.0 / 16.0)),
            .gradient = @splat(@splat(0)),
            .diff_a = @splat(0),
            .diff_b = @splat(0),
            .delta = @splat(0),
            .parents = @splat(.{ 0, 0 }),
            .value = @splat(0),
        };

        /// Evaluates the network.
        pub fn eval(net: *Self, x: *const [dim_in]f32) *const [dim_out]f32 {
            @setFloatMode(.optimized);
            // Note: One single for-loop is faster, however, this structure is
            // in preparation for multithreading.
            // This inline for loop avoids two separate for loops with the same body
            // except input being different for the first layer, nodes from 0 to shape[i].
            inline for (layer_ranges, 0..) |range, l| {
                const input = if (l == 0) x else net.value;
                for (range.from..range.to()) |j| {
                    const parents = net.parents[j];
                    const a = input[parents[0]];
                    const b = input[parents[1]];
                    net.value[j] = @reduce(.Add, net.sigma[j] * Gate.vector(a, b));
                }
            }
            return @ptrCast(net.value[layer_last.from..layer_last.to()]);
        }

        /// Evaluates the network and caches the relevant data for the
        /// backwardPass: del_a, del_b, and the gradient, which is later correctly
        /// scaled during the backward pass.
        pub fn forwardPass(net: *Self, input: *const [dim_in]f32) *const [dim_out]f32 {
            @setFloatMode(.optimized);
            @memset(&net.gradient, @splat(0));
            // See the comment in eval(...) above for why we loop in this manner.
            inline for (layer_ranges, 0..) |range, l| {
                const parent_value = if (l == 0) input else net.value;
                for (range.from..range.to()) |j| {
                    const parents = net.parents[j];
                    const a = parent_value[parents[0]];
                    const b = parent_value[parents[1]];
                    const sigma = net.sigma[j];
                    net.diff_a[j] = @reduce(.Add, sigma * Gate.diff_a(b));
                    net.diff_b[j] = @reduce(.Add, sigma * Gate.diff_b(a));
                    const gate = Gate.vector(a, b);
                    net.value[j] = @reduce(.Add, sigma * gate);
                    net.gradient[j] += sigma * (gate - @as(f32x16, @splat(net.value[j])));
                }
            }
            return @ptrCast(net.value[layer_last.from..layer_last.to()]);
        }

        /// Accumulates the loss gradient to the cost gradient with respect to the network's logits.
        /// One must first call forwardPass to cache the relevant data.
        pub fn backwardPass(net: *Self, cost_gradient: *[node_count]f32x16) void {
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
                for (range.from..range.to()) |j| {
                    const delta = net.delta[j];
                    cost_gradient[j] += @as(f32x16, @splat(tau * delta)) * net.gradient[j];
                    if (l != 0) {
                        const parents = net.parents[j];
                        net.delta[parents[0]] += 2.0 * delta * net.diff_a[j];
                        net.delta[parents[1]] += 2.0 * delta * net.diff_b[j];
                    }
                }
            }
        }

        /// Returns the delta of the last layer.
        pub fn lastLayerDelta(net: *Self) *[layer_last.len]f32 {
            return net.delta[layer_last.from..];
        }

        /// Updates the probability distribution, sigma, as softmax of the given logits.
        pub fn setLogits(net: *Self, logits: *[node_count]f32x16) void {
            for (&net.sigma, logits) |*sigma, logit| {
                sigma.* = if (options.softmax_base_2) softmax2(logit) else softmax(logit);
            }
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

// Fixme: Add FloatType as an option.
pub const ModelOptions = struct {
    shape: []const usize,
    InputLayer: ?fn (usize) type = null,
    OutputLayer: ?fn (usize) type = null,
    Optimizer: ?fn (usize) type = null,
    Loss: ?type = null,
};

/// A model encapsulates a differentiable logic gate network into a machine learning context.
/// It allows you to specify, most importantly, a loss function and optimizer.
pub fn Model(comptime options: ModelOptions) type {
    return struct {
        const Self = @This();

        // Unpack options.
        const shape = options.shape;
        const NetworkType = Network(.{ .shape = shape });
        const parameter_count = NetworkType.parameter_count;

        const InputLayer = if (options.InputLayer) |IL| IL(NetworkType.dim_in) else input_layer.Identity(NetworkType.dim_in);
        const OutputLayer = if (options.OutputLayer) |OL| OL(NetworkType.dim_out) else output_layer.Identity(NetworkType.dim_out);
        const Optimizer = if (options.Optimizer) |O| O(parameter_count) else optim.Adam(.default)(parameter_count);
        const Loss = if (options.Loss) |C| C else loss_function.HalvedMeanSquareError(OutputLayer.dim_out);

        const Input = InputLayer.Type;
        const Output = OutputLayer.Type;
        pub const Datapoint = struct {
            input: Input,
            output: Output,
        };

        // The parameters of the network, parameters 0 ... network.parameter_count are reserved for the network,
        // the remaining are for the output layer, if any.
        parameters: [parameter_count]f32 align(64),
        // The gradient of the loss function with resepct to every parameter for
        // a single datapoint, or cost for a full dataset.
        gradient: [parameter_count]f32 align(64),

        network: Network(.{
            .shape = shape,
        }),
        layer_input: InputLayer,
        layer_output: OutputLayer,
        optimizer: Optimizer,

        pub const default = Self{
            .parameters = @bitCast(NetworkType.passtrough_biased),
            .gradient = @splat(0),
            .network = .default,
            .layer_input = undefined,
            .layer_output = undefined,
            .optimizer = .default,
        };

        /// Returns the mean loss over a dataset.
        pub fn cost(model: *Self, dataset: []const Datapoint) f32 {
            assert(dataset.len > 0);
            var result: f32 = 0;
            for (dataset) |point| result += model.loss(&point.input, &point.output);
            return result / @as(f32, @floatFromInt(dataset.len));
        }

        /// Computes the gradient of the sum of the loss function over the entire dataset.
        pub fn differentiate(model: *Self, dataset: []const Datapoint) void {
            @memset(&model.gradient, 0);
            // Fixme: Branch on dataset length for optimal network evaluation during regular SGD.
            model.network.setLogits(@ptrCast(&model.parameters));
            for (dataset) |data| {
                const input = &data.input;
                const prediction = model.forwardPass(input);
                const actual = &data.output;
                model.backwardPass(prediction, actual);
            }
        }

        /// Accumulates the gradient backwards.
        pub fn backwardPass(model: *Self, prediction: *const Output, actual: *const Output) void {
            Loss.gradient(prediction, actual, &model.layer_output.delta);
            model.layer_output.backwardPass(model.network.lastLayerDelta());
            model.network.backwardPass(@ptrCast(&model.gradient));
        }

        /// Evaluates the model, caching all necessary data to compute the gradient in the backward pass.
        pub fn forwardPass(model: *Self, input: *const Input) *const Output {
            const a = model.layer_input.eval(input);
            const b = model.network.forwardPass(a);
            return model.layer_output.forwardPass(b);
        }

        /// The loss of the model given some input and its expected output.
        pub fn loss(model: *Self, input: *const Input, actual: *const Output) f32 {
            return Loss.eval(model.eval(input), actual);
        }

        /// Evaluates the model.
        pub fn eval(model: *Self, input: *const Input) *const Output {
            const a = model.layer_input.eval(input);
            const b = model.network.eval(a);
            return model.layer_output.eval(b);
        }

        /// Trains the model on a given dataset for a specified amount of epochs and batch size.
        /// Every epoch the model is 'validated' on another given dataset.
        pub fn train(model: *Self, training: []const Datapoint, validate: []const Datapoint, epoch_count: usize, batch_size: usize) void {
            assert(batch_size > 0);
            for (0..epoch_count) |epoch| {
                var offset: usize = 0;
                while (training.len > offset) : (offset += batch_size) {
                    const subset = training[offset..@min(offset + batch_size, training.len)];
                    model.differentiate(subset);
                    model.optimizer.step(&model.parameters, &model.gradient);
                }
                if (validate.len > 0) std.debug.print("epoch: {d}\tloss: {d}\n", .{ epoch, model.cost(validate) });
            }
        }
    };
}
