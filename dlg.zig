const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const EnumArray = std.EnumArray;
const MultiArrayList = std.MultiArrayList;
const ArrayList = std.ArrayList;
const AutoArrayHashMap = std.AutoArrayHashMap;

pub const v16f32 = @Vector(16, f32);

const SoftGate = struct {
    /// Returns a vector containg representing all values of a soft logic gate.
    pub fn vector(a: f32, b: f32) v16f32 {
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

    /// Returns a the soft gate vector differentiated by the first variable.
    /// Note that it only depends on the second.
    pub fn del_a(b: f32) v16f32 {
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

    pub fn del_b(a: f32) v16f32 {
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

pub fn softmax(w: v16f32) v16f32 {
    const sigma = @exp(w);
    const denom = @reduce(.Add, sigma);
    return sigma / @as(v16f32, @splat(denom));
}

// Fixme: Make float type an optional parameter.
pub const Network = struct {
    // Indices into the nodes slice.
    const NodeIndex = u32;
    // Currently the graph is stored as a list of lists.
    // Consider using a compressed sparse row format instead.
    const Node = struct {
        // Note: One can make these relative to each layer; this will save memory.
        // For example if each layer at most has 60_000 nodes, then u16 suffices
        // as a relative index type.
        parents: [2]NodeIndex,
        children: []NodeIndex,

        weights: v16f32,
        // The gradient of the cost function with respect to the weights of the node.
        gradient: v16f32,
        // The current feedforward value.
        value: f32 = 0,
        // Used for backpropagation, defined as dC/dvalue,
        // where C is the cost function. This in turn is used to compute the gradient.
        delta: f32,

        // Adam optimizer data.
        adam_v: v16f32,
        adam_m: v16f32,

        pub fn eval(node: Node, a: f32, b: f32) f32 {
            const sigma = softmax(node.weights);
            return @reduce(.Add, sigma * SoftGate.vector(a, b));
        }

        // Returns the derivative of eval with respect to the first parent.
        pub fn del_a(node: Node, b: f32) f32 {
            const sigma = softmax(node.weights);
            return @reduce(.Add, sigma * SoftGate.del_a(b));
        }

        // Returns the derivative of eval with respect to the second parent.
        pub fn del_b(node: Node, a: f32) f32 {
            const sigma = softmax(node.weights);
            return @reduce(.Add, sigma * SoftGate.del_b(a));
        }

        pub fn eval_(weights: v16f32, a: f32, b: f32) f32 {
            const sigma = softmax(weights);
            return @reduce(.Add, sigma * SoftGate.vector(a, b));
        }

        pub fn del_a_(weights: v16f32, b: f32) f32 {
            const sigma = softmax(weights);
            return @reduce(.Add, sigma * SoftGate.del_a(b));
        }

        pub fn del_b_(weights: v16f32, a: f32) f32 {
            const sigma = softmax(weights);
            return @reduce(.Add, sigma * SoftGate.del_b(a));
        }
    };

    input_dim: usize,
    layers: []MultiArrayList(Node).Slice,

    pub fn lastLayer(net: Network) MultiArrayList(Node).Slice {
        return net.layers[net.layers.len - 1];
    }

    pub fn eval(net: Network, x: []const f32) void {
        assert(x.len == net.input_dim);

	// Note: Two for loops is faster than a single with a branch.
        const first = net.layers[0];
        for (first.items(.value), first.items(.parents), first.items(.weights)) |*value, parents, weights| {
            const a = x[parents[0]];
            const b = x[parents[1]];
            value.* = Node.eval_(weights, a, b);
        }

        for (net.layers[1..], net.layers[0 .. net.layers.len - 1]) |layer, prev| {
            const prev_values = prev.items(.value);
            for (layer.items(.value), layer.items(.parents), layer.items(.weights)) |*value, parents, weights| {
                const a = prev_values[parents[0]];
                const b = prev_values[parents[1]];
                value.* = Node.eval_(weights, a, b);
            }
        }
    }
    
    // Compute the delta for each node relative to a given datapoint.
    // See the definition of the Node struct.
    // Fixme: Currently assumes the network is trained with a halved mean square error.
    pub fn backprop(net: Network, x: []const f32, y: []const f32) void {

        const last = net.lastLayer();
        assert(x.len == net.input_dim);
        assert(y.len == last.len);

        net.eval(x);

        // Compute the delta for the last layer.
        for (last.items(.delta), last.items(.value), y) |*delta, value, y_value| {
            // Fixme: Hardcoded gradient of the halved mean square error.
            delta.* = value - y_value;
        }
        // Compute the delta of the previous layers, back to front.
        for (2..net.layers.len + 1) |i| {
            const layer = net.layers[net.layers.len - i];
            const after = net.layers[net.layers.len + 1 - i];
            const values = layer.items(.value);
            for (layer.items(.delta), layer.items(.children), 0..) |*delta, children, parent_index| {
                delta.* = 0;
                for (children) |child_index| {
                    // Note: Benchmark unpacking single fields here instead of the entire child.
                    const child = after.get(child_index);
                    const a = values[child.parents[0]];
                    const b = values[child.parents[1]];
                    delta.* += child.delta * if (child.parents[0] == parent_index) child.del_a(b) else child.del_b(a);
                }
            }
        }
    }

    // Updates the gradient of the network for a given datapoint.
    // Fixme: Now the gradient of softmax is hardcoded.
    // Fixme: Move the inner 2 for loops to a separate function.
    pub fn update_gradient(net: Network, x: []const f32, y: []const f32) void {
        assert(x.len == net.input_dim);
        assert(y.len == net.lastLayer().len);

        net.backprop(x, y);

        for (net.layers, 0..) |layer, i| {
            // This branch does not have a huge impact, but consider splitting into two loops.
            const prev_values = if (i == 0) x else net.layers[i - 1].items(.value);
            for (layer.items(.parents), layer.items(.weights), layer.items(.gradient), layer.items(.delta)) |parents, weights, *gradient, delta| {
                const a = prev_values[parents[0]];
                const b = prev_values[parents[1]];
                const gate = SoftGate.vector(a, b);
                const sigma = softmax(weights);
                const e1: v16f32 = .{1} ++ .{0} ** 15;
                var result: [16]f32 = .{0} ** 16;
                inline for (&result, 0..) |*partial, k| {
                    const kronecker = std.simd.rotateElementsRight(e1, k);
                    const sigma_k: v16f32 = @splat(@reduce(.Add, kronecker * sigma));
                    partial.* += @reduce(.Add, sigma * gate * (kronecker - sigma_k) * @as(v16f32, @splat(delta)));
                }
                gradient.* += result;
            }
        }
    }

    pub fn deinit(net: Network, allocator: Allocator) void {
        for (net.nodes) |node| allocator.free(node.children);
        allocator.free(net.nodes);
        allocator.free(net.shape);
    }

    /// Initializes a network with specified shape with random connections inbetween.
    /// Each node is guaranteed to have at least one child.
    /// Note that shape[0] is the input dimension and shape[shape.len - 1] is the output dimension.
    pub fn initRandom(allocator: Allocator, shape: []const usize) !Network {
        // Shape must at least specify input and output dimensions.
        assert(shape.len >= 2);
        // Assert that the network does not shrink by more than a factor of 2,
        // this forces some nodes to have no children, which causes those nodes to become useless.
        for (shape[0 .. shape.len - 1], shape[1..]) |dim, next| {
            assert(next * 2 >= dim);
        }

        var prng = std.Random.DefaultPrng.init(blk: {
            var seed: u64 = undefined;
            try std.posix.getrandom(std.mem.asBytes(&seed));
            break :blk seed;
        });
        const rand = prng.random();

        var net: Network = undefined;
        net.input_dim = shape[0];
        net.layers = try allocator.alloc(MultiArrayList(Node).Slice, shape.len - 1);
        for (net.layers, shape[1..]) |*layer, dim| {
            var tmp: MultiArrayList(Node) = .{};
            try tmp.resize(allocator, dim);
            layer.* = tmp.slice();
        }

        // Initialize nodes.
        for (net.layers) |*layer| {
            for (layer.items(.weights)) |*weights| {
                // Initialize weights to be biased toward the pass through gates.
                weights.* = [_]f32{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10;
            }
            @memset(layer.items(.gradient), [_]f32{0} ** 16);
            @memset(layer.items(.children), &.{});
            @memset(layer.items(.adam_m), [_]f32{0} ** 16);
            @memset(layer.items(.adam_v), [_]f32{0} ** 16);
        }

	// Initialize node parents such that each node has at least one child, excluding the last layer.
        var stack = try ArrayList(usize).initCapacity(allocator, std.mem.max(usize, shape));
        defer stack.deinit();
        for (net.layers, shape[0..shape.len - 1]) |layer, prev_dim| {
            // Fill the stack with all possible indices of the previous layer in random order.
       	    for (0..prev_dim) |i| stack.appendAssumeCapacity(i);
	    rand.shuffle(usize, stack.items);

            for (layer.items(.parents)) |*parents| {
                const first = stack.pop() orelse rand.uintLessThan(usize, prev_dim);
                const second = stack.pop() orelse while (true) {
                    const i = rand.uintLessThan(usize, prev_dim);
                    // Fixme: Will fail if dim == 1.
                    if (i != first) break i;
                } else unreachable;
                parents.* = .{ @truncate(first), @truncate(second) };
            }
        }
	// Find the children of each node.
        var buffer = ArrayList(NodeIndex).init(allocator);
        defer buffer.deinit();
        for (net.layers[0 .. net.layers.len - 1], net.layers[1..]) |*layer, next| {
            for (layer.items(.children), 0..) |*children, parent_index| {
                for (next.items(.parents), 0..) |parents, child_index| {
                    if (parents[0] == parent_index or parents[1] == parent_index) try buffer.append(@truncate(child_index));
                }
                children.* = try buffer.toOwnedSlice();
            }
        }
        
        // Assert parent indices and that every node has at least one child excepting the ones in the last layer.
        for (net.layers, shape[0..shape.len - 1], shape[1..], 0..) |layer, dim_prev, dim_next, i| {
        	for (layer.items(.parents), layer.items(.children)) |parents, children| {
        		assert(parents[0] < dim_prev);
        		assert(parents[1] < dim_prev);
        		assert(children.len > 0 or i == net.layers.len - 1);
        		for (children) |child_index| assert(child_index < dim_next);
        	}
        }
        // Assert child indices.
        for (net.layers[0 .. net.layers.len - 1], net.layers[1..]) |layer, next| {
        	for (layer.items(.children), 0..) |children, parent_index| {
        		for (children) |child_index| {
        			const parents = next.items(.parents)[child_index];
        			assert(parents[0] == parent_index or parents[1] == parent_index);
        		}
        	}
        }
        
        return net;
    }
};
