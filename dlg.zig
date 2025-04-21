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
pub const f32x8 = @Vector(8, f32);

/// Ranges are stored in this manner to make the illegal ranges (where to < from) unrepresentable.
const Range = struct {
    from: usize,
    len: usize,

    pub fn to(range: Range) usize {
        return range.from + range.len;
    }
};

pub fn DoubleBuffer(T: type, len: usize) type {
    return struct {
        const Self = @This();

        data: [2 * len]T,
        half: u1,

        pub fn init(element: T) Self {
            return Self{
                .data = @splat(element),
                .half = 0,
            };
        }

        pub fn front(buffer: *const Self) *const [len]T {
            const from = buffer.half * len;
            const to = from + len;
            return @ptrCast(buffer.data[from..to]);
        }

        pub fn front_slice(buffer: *const Self, comptime slice_len: usize) *const [slice_len]T {
            return @ptrCast(buffer.front()[0..slice_len]);
        }

        pub fn back(buffer: *Self) *[len]T {
            const from = (buffer.half +% 1) * len;
            const to = from + len;
            return @ptrCast(buffer.data[from..to]);
        }

        pub fn back_slice(buffer: *Self, comptime slice_len: usize) *[slice_len]T {
            return @ptrCast(buffer.back()[0..slice_len]);
        }

        pub fn swap(buffer: *Self) void {
            buffer.half +%= 1;
        }
    };
}

/// Logits are normalized such that max(logits) == 0.
/// When the logits grow to big exp^logit will explode.
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
    /// Only intended for reference. The current implementation
    /// inlines this construction in forwardPass.
    pub fn vector(a: f32, b: f32) f32x16 {
        return .{
            0, // false
            a * b, // and
            a - a * b, // a and not b
            a, // passthrough a
            b - a * b, // b and not a
            b, // passthrough b
            a + b - 2 * a * b, // xor
            a + b - a * b, // xnor
            // The following values are simply negation of the above ones in order.
            // Many authors reverse this order, however this leads to a less efficient
            // SIMD construction of the vector.
            1,
            1 - a * b,
            1 - (a - a * b),
            1 - a,
            1 - (b - a * b),
            1 - b,
            1 - (a + b - 2 * a * b),
            1 - (a + b - a * b),
        };
    }

    /// Returns the soft gate vector differentiated by the first variable.
    /// Note that it only depends on the second variable.
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
            0,
            0,
            -b,
            -(1 - b),
            -1,
            b,
            0,
            -(1 - 2 * b),
            0,
        };
    }

    /// Returns the soft gate vector differentiated by the second variable.
    /// Note that it only depends on the first variable.
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
            0,
            -a,
            a,
            0,
            -(1 - a),
            -1,
            -(1 - 2 * a),
            -(1 - a),
        };
    }
};

const NetworkOptions = struct {
    shape: []const usize,
    softmax_base_2: bool = false,
    gate_temperature: f32 = 1.0,
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
        pub const passtrough_biased: [node_count]f32x16 = @splat(.{ 0, 0, 0, 16, 0, 16 } ++ .{0} ** 10);

        // The network does not have a layer with shape[0] number of nodes. But rather the first layer,
        // nodes from 0 .. shape[1] will have parent indices that range from 0 .. shape[0], which is
        // the dimension of the input.
        const dim_in = shape[0];
        // The last layer
        const dim_out = shape[shape.len - 1];
        comptime {
            assert(dim_out == layer_last.len);
        }

        const NodeIndex = std.math.IntFittingRange(0, std.mem.max(usize, shape));

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
        diff: [node_count][2]f32 align(64),

        // The indices of the node's parents, for the first layer i.e. nodes 0..shape[1] these
        // are instead indices into the input.
        // Note: These can be entirely optimized away. Using a reversible LCG one can generate
        // the indices using a fixed seed. Additionally one can safely assume that the j:th
        // node in the l:th layer is the parent of the j:th (possibly wrapping) node in l+1:th layer.
        // In this way we guarantee that each node has children. To do generally will be quite tricky,
        // especially if avoiding modulo operations. Instead of n % range, one can do use a multiply
        // followed by a shift. See the blog post: https://lemire.me/blog/2016/06/30/fast-random-shuffling/
        // It is worth the effort, since then the data structure can be truly static, no dynamic initialization.
        parents: [node_count][2]NodeIndex align(64),

        // A default network with uniform probabilities for each gate.
        // Before using the network on must call randomizeConnections to
        // initialize every node's parents.
        pub const default = Self{
            .sigma = @splat(@splat(1.0 / 16.0)),
            .gradient = @splat(@splat(0)),
            .diff = @splat(.{ 0, 0 }),
            .parents = @splat(.{ 0, 0 }),
        };

        /// Evaluates the network.
        pub fn eval(net: *Self, buffer: anytype) *const [dim_out]f32 {
            @setFloatMode(.optimized);
            // This inline for loop avoids two separate for loops with the same body
            // except input being different for the first layer, nodes from 0 to shape[i].
            inline for (layer_ranges, 0..) |range, l| {
                const inputs = buffer.front_slice(shape[l]);
                const activations = buffer.back_slice(range.len);
                for (activations, range.from..range.to()) |*activation, j| {
                    const parents = net.parents[j];
                    const a = inputs[parents[0]];
                    const b = inputs[parents[1]];
                    activation.* = @reduce(.Add, net.sigma[j] * Gate.vector(a, b));
                }
                buffer.swap();
            }
            return buffer.front_slice(layer_last.len);
        }

        /// Evaluates the network and caches the relevant data for the
        /// backwardPass: diff and the gradient, which is later correctly
        /// scaled during the backward pass.
        pub fn forwardPass(net: *Self, buffer: anytype) *const [dim_out]f32 {
            @setFloatMode(.optimized);
            // See the comment in eval(...) above for why we loop in this manner.
            inline for (layer_ranges, 0..) |range, l| {
                const inputs = buffer.front_slice(shape[l]);
                const activations = buffer.back_slice(range.len);
                for (activations, range.from..range.to()) |*activation, j| {
                    const parents = net.parents[j];
                    const a = inputs[parents[0]];
                    const b = inputs[parents[1]];
                    const sigma = net.sigma[j];

                    // Inline construction of Gate.vector(a, b), Gate.diff_a(b), and Gate.diff_b(a).
                    // We do it in this way since all three depend on mix_coef, and two out of three
                    // depend on a_coef/b_coef.
                    const a_coef: f32x8 = .{ 0, 0, 1, 1, 0, 0, 1, 1 };
                    const b_coef: f32x8 = .{ 0, 0, 0, 0, 1, 1, 1, 1 };
                    const mix_coef: f32x8 = .{ 0, 1, -1, 0, -1, 0, -2, -1 };

                    // All three desired vectors have halfway symmetry so we only
                    // explicitly construct the first half.
                    const diff_a_half = a_coef + mix_coef * @as(f32x8, @splat(b));
                    const diff_b_half = b_coef + mix_coef * @as(f32x8, @splat(a));
                    net.diff[j] = .{
                        @reduce(.Add, sigma * std.simd.join(diff_a_half, -diff_a_half)),
                        @reduce(.Add, sigma * std.simd.join(diff_b_half, -diff_b_half)),
                    };
                    const gate_half =
                        a_coef * @as(f32x8, @splat(a)) +
                        b_coef * @as(f32x8, @splat(b)) +
                        mix_coef * @as(f32x8, @splat(a * b));
                    const gate = std.simd.join(gate_half, @as(f32x8, @splat(1)) - gate_half);

                    activation.* = @reduce(.Add, sigma * gate);
                    net.gradient[j] = sigma * (gate - @as(f32x16, @splat(activation.*)));
                }
                buffer.swap();
            }
            return buffer.front_slice(layer_last.len);
        }

        /// Accumulates the loss gradient to the cost gradient with respect to the network's logits.
        /// One must first call forwardPass to cache the relevant data.
        pub fn backwardPass(net: *Self, cost_gradient: *[node_count]f32x16, buffer: anytype) void {
            @setFloatMode(.optimized);

            // Compensating factor for the softmax temperature.
            const tau = (if (options.softmax_base_2) @log(2.0) else 1.0) / options.gate_temperature;

            // Propagate the delta backwards, layer by layer, and use it to compute the gradient.
            // Inlining this loop ensures that we avoid the inner branch that is only valid for
            // the first layer.
            comptime var l: usize = layer_count;
            inline while (l > 0) {
                l -= 1;
                const range = layer_ranges[l];
                const child_deltas = buffer.front_slice(range.len);
                const parent_deltas = buffer.back_slice(shape[l]);
                @memset(parent_deltas, 0);
                for (child_deltas, range.from..range.to()) |delta, j| {
                    cost_gradient[j] += @as(f32x16, @splat(tau * delta)) * net.gradient[j];
                    if (l != 0) {
                        const parents = net.parents[j];
                        parent_deltas[parents[0]] += delta * net.diff[j][0];
                        parent_deltas[parents[1]] += delta * net.diff[j][1];
                    }
                }
                buffer.swap();
            }
        }

        /// Returns the delta of the last layer.
        pub fn lastLayerDelta(net: *Self) *[layer_last.len]f32 {
            return net.delta[layer_last.from..];
        }

        /// Updates the probability distribution, sigma, as softmax of the given logits.
        pub fn setLogits(net: *Self, logits: *[node_count]f32x16) void {
            for (&net.sigma, logits) |*sigma, logit| {
                sigma.* = if (options.softmax_base_2) softmax2(logit / @as(f32x16, @splat(options.gate_temperature))) else softmax_approx(logit / @as(f32x16, @splat(options.gate_temperature)));
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
                _ = offset;
                var i: usize = 0;
                for (net.parents[range.from..range.to()]) |*parents| {
                    const first = i % dim_prev;
                    const second = (i + 1) % dim_prev;
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
    gate_temperature: f32 = 1,
};

const NetworkOptions = struct {
	Layers: []const type,
};

pub fn Network(options: NetworkOptions) type {
	layers: std.meta.Tuple(options.Layers);
	const layer_count = options.Layers.len;
	const parameter_count = blk: {
		var result: usize = 0;
		inline for (std.meta.Tuple(options.Layers)) |Layer| result += Layer.dim;
	};
	
	buffer: DoubleBuffer(f32, 

	pub fn forwardPass
	pub fn backwardPass
	pub fn preprocess(*[parameter_count]
};

const Layer = struct {
	const dim_in: usize;
	const dim_out: usize;
	const parameter_count: usize;

	const Self = @This();
	pub fn eval(self: *Self, 
	
	pub fn preProcess(self: *Self, *[parameter_count]f32: parameters) void {
	
	}
	
	pub fn postProcess(self: *Self, *[parameter_count]f32: parameters) void {
	
	}
	
	pub fn eval(self: *Self,
};

/// A model encapsulates a differentiable logic gate network into a machine learning context.
/// It allows you to specify, most importantly, a loss function and optimizer.
pub fn Model(comptime options: ModelOptions) type {
    return struct {
        const Self = @This();

        // Unpack options.
        const shape = options.shape;
        const NetworkType = Network(.{
            .shape = shape,
            .gate_temperature = options.gate_temperature,
        });
        const parameter_count = NetworkType.parameter_count;

        const InputLayer = if (options.InputLayer) |IL| IL(NetworkType.dim_in) else input_layer.Identity(NetworkType.dim_in);
        const OutputLayer = if (options.OutputLayer) |OL| OL(NetworkType.dim_out) else output_layer.Identity(NetworkType.dim_out);
        const Optimizer = if (options.Optimizer) |O| O(parameter_count) else optim.Adam(.default)(parameter_count);
        const Loss = if (options.Loss) |C| C else loss_function.HalvedMeanSquareError(OutputLayer.dim_out);

        const Feature = InputLayer.Feature;
        const Prediction = OutputLayer.Prediction;
        const Label = Loss.Label;
        pub const Dataset = struct {
            features: []const Feature,
            labels: []const Label,

            pub const empty = Dataset{ .features = &.{}, .labels = &.{} };

            pub fn len(dataset: Dataset) usize {
                assert(dataset.features.len == dataset.labels.len);
                return dataset.features.len;
            }

            pub fn init(features: []const Feature, labels: []const Label) Dataset {
                assert(features.len == labels.len);
                return .{
                    .features = features,
                    .labels = labels,
                };
            }

            pub fn slice(dataset: Dataset, from: usize, to: usize) Dataset {
                assert(from <= to);
                return .{
                    .features = dataset.features[from..to],
                    .labels = dataset.labels[from..to],
                };
            }
        };
        const dim_max = @max(std.mem.max(usize, shape), OutputLayer.dim_out);

        // The parameters of the network, parameters 0 ... network.parameter_count are reserved for the network,
        // the remaining are for the output layer, if any.
        parameters: [parameter_count]f32 align(64),
        // The gradient of the loss function with resepct to every parameter for
        // a single datapoint, or cost for a full dataset.
        gradient: [parameter_count]f32 align(64),

        network: Network(.{
            .shape = shape,
        }),
        optimizer: Optimizer,
        buffer: DoubleBuffer(f32, dim_max),

        pub const default = Self{
            .parameters = @bitCast(NetworkType.passtrough_biased),
            .gradient = @splat(0),
            .network = .default,
            .optimizer = .default,
            .buffer = .init(0),
        };

        /// Returns the mean loss over a dataset.
        pub fn cost(model: *Self, dataset: Dataset) f32 {
            assert(dataset.len() > 0);
            var result: f32 = 0;
            for (dataset.features, dataset.labels) |feature, label| result += model.loss(&feature, &label);
            return result / @as(f32, @floatFromInt(dataset.len()));
        }

        /// Computes the gradient of the sum of the loss function over the entire dataset.
        pub fn differentiate(model: *Self, dataset: Dataset) void {
            @memset(&model.gradient, 0);
            // Fixme: Branch on dataset length for optimal network evaluation during regular SGD.
            model.network.setLogits(@ptrCast(&model.parameters));
            for (dataset.features, dataset.labels) |feature, label| {
                const prediction = model.forwardPass(&feature);
                model.backwardPass(prediction, &label);
            }
        }

        /// Accumulates the gradient backwards.
        pub fn backwardPass(model: *Self, prediction: *const Prediction, label: *const Label) void {
            const buffer = &model.buffer;
            Loss.gradient(prediction, label, buffer);
            OutputLayer.backwardPass(buffer);
            model.network.backwardPass(@ptrCast(&model.gradient), buffer);
        }

        /// Evaluates the model, caching all necessary data to compute the gradient in the backward pass.
        pub fn forwardPass(model: *Self, feature: *const Feature) *const Prediction {
            const buffer = &model.buffer;
            InputLayer.eval(feature, buffer);
            _ = model.network.forwardPass(buffer);
            const prediction = OutputLayer.forwardPass(buffer);
            return prediction;
        }

        /// The loss of the model given some input and its expected output.
        pub fn loss(model: *Self, feature: *const Feature, label: *const Label) f32 {
            return Loss.eval(model.eval(feature), label);
        }

        /// Evaluates the model.
        pub fn eval(model: *Self, feature: *const Feature) *const Prediction {
            const buffer = &model.buffer;
            InputLayer.eval(feature, buffer);
            _ = model.network.eval(buffer);
            return OutputLayer.eval(buffer);
        }

        /// Trains the model on a given dataset for a specified amount of epochs and batch size.
        /// Every epoch the model is 'validated' on another given dataset.
        pub fn train(model: *Self, training: Dataset, validate: Dataset, epoch_count: usize, batch_size: usize) void {
            assert(batch_size > 0);
            for (0..epoch_count) |epoch| {
                var offset: usize = 0;
                while (training.len() > offset) : (offset += batch_size) {
                    const subset = training.slice(offset, @min(offset + batch_size, training.len()));
                    model.differentiate(subset);
                    model.optimizer.step(&model.parameters, &model.gradient);
                }
                if (validate.len() > 0) std.debug.print("epoch: {d}\tloss: {d}\n", .{ epoch, model.cost(validate) });
            }
        }
    };
}
