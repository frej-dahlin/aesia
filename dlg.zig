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
    };

    // Fixme: Use a MultiArrayList instead (aka SoA), this would be much more cache-friendly.
    // The forward pass only needs the nodes parents, while the backward pass
    // does not need the parent indices.
    nodes: []Node,
    // The shape of the network, shape[0] and shape[shape.len - 1] are the
    // input and output dimensions, respectively. For l >= 1, shape[l] is
    // the dimension of layer l.
    // Note: With the above relative index changes this will need to become a slice of offsets into the separate layers.
    shape: []usize,

    pub fn lastLayer(network: Network) []Node {
        return network.nodes[network.nodes.len - network.shape[network.shape.len - 1] ..];
    }

    pub fn eval(network: Network, x: []const f32) void {
        assert(x.len == network.shape[0]);

        const nodes = network.nodes;
        const shape = network.shape;

        // Evaluate first layer.
        for (nodes[0..shape[1]]) |*node| {
            const a = x[node.parents[0]];
            const b = x[node.parents[1]];
            node.value = node.eval(a, b);
        }
        // Evaluate the rest of the network.
        for (nodes[shape[1]..]) |*node| {
            const a = nodes[node.parents[0]].value;
            const b = nodes[node.parents[1]].value;
            node.value = node.eval(a, b);
        }
    }

    pub fn output(net: Network, out: []f32) void {
        assert(out.len == net.lastLayer().len);
        for (net.lastLayer(), out) |node, *y| y.* = node.value;
    }

    // Compute the delta for each node relative to a given datapoint.
    // See the definition of the Node struct.
    // Fixme: Currently assumes the network is trained with a halved mean square error.
    pub fn backprop(network: Network, x: []const f32, y: []const f32) void {
        assert(x.len == network.shape[0]);
        assert(y.len == network.shape[network.shape.len - 1]);

        network.eval(x);

        const nodes = network.nodes;
        const shape = network.shape;

        // Compute the delta of the last layer.
        const last_layer = network.lastLayer();
        for (last_layer, 0..) |*node, j| {
            // Fixme: Hardcoded gradient of the halved mean square error.
            node.delta = node.value - y[j];
        }

        // Compute the delta of the previous layers, back to front.
        var offset = nodes.len - last_layer.len;
        for (2..shape.len) |i| {
            const from = offset - shape[shape.len - i];
            const to = offset;
            for (nodes[from..to], 0..) |*node, j| {
                node.delta = 0;
                for (node.children) |child_index| {
                    const child = nodes[child_index];
                    const a = nodes[child.parents[0]].value;
                    const b = nodes[child.parents[1]].value;
                    node.delta += child.delta * if (child.parents[0] == j)
                        child.del_a(b)
                    else
                        child.del_b(a);
                }
            }
            offset = from;
        }
    }

    // Updates the gradient of the network for a given datapoint.
    // Fixme: Now the gradient of softmax is hardcoded.
    // Fixme: Move the inner 2 for loops to a separate function.
    pub fn update_gradient(network: Network, x: []const f32, y: []const f32) void {
        assert(x.len == network.shape[0]);
        assert(y.len == network.shape[network.shape.len - 1]);

        network.backprop(x, y);
        const nodes = network.nodes;

        // Note: Can use inline for-loops to contract these into one.
        for (nodes[0..network.shape[1]]) |*node| {
            const a = x[node.parents[0]];
            const b = x[node.parents[1]];
            const sigma = softmax(node.weights);
            const gate = SoftGate.vector(a, b);
            const e1: v16f32 = .{1} ++ .{0} ** 15;
            var gradient: [16]f32 = .{0} ** 16;
            inline for (&gradient, 0..) |*partial, k| {
                const kronecker = std.simd.rotateElementsRight(e1, k);
                const sigma_k: v16f32 = @splat(@reduce(.Add, kronecker * sigma));
                partial.* += @reduce(.Add, sigma * gate * (kronecker - sigma_k) * @as(v16f32, @splat(node.delta)));
            }
            node.gradient = gradient;
        }
        for (nodes[network.shape[1]..]) |*node| {
            const a = nodes[node.parents[0]].value;
            const b = nodes[node.parents[1]].value;
            const sigma = softmax(node.weights);
            const gate = SoftGate.vector(a, b);
            const e1: v16f32 = .{1} ++ .{0} ** 15;
            var gradient: [16]f32 = .{0} ** 16;
            inline for (&gradient, 0..) |*partial, k| {
                const kronecker = std.simd.rotateElementsRight(e1, k);
                const sigma_k: v16f32 = @splat(@reduce(.Add, kronecker * sigma));
                partial.* += @reduce(.Add, sigma * gate * (kronecker - sigma_k) * @as(v16f32, @splat(node.delta)));
            }
            node.gradient = gradient;
        }
    }

    pub fn deinit(network: Network, allocator: Allocator) void {
        for (network.nodes) |node| allocator.free(node.children);
        allocator.free(network.nodes);
        allocator.free(network.shape);
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

        var network: Network = undefined;

        // Copy shape slice.
        network.shape = try allocator.alloc(usize, shape.len);
        @memcpy(network.shape, shape);
        errdefer allocator.free(network.shape);

        // Count number of nodes in the network, this excludes the input layer.
        var node_count: usize = 0;
        for (shape[1..]) |dim| node_count += dim;
        network.nodes = try allocator.alloc(Node, node_count);
        errdefer allocator.free(network.nodes);

        // Temporary slice of the layers.
        const layers: [][]Node = try allocator.alloc([]Node, shape.len - 1);
        defer allocator.free(layers);
        var offset: usize = 0;
        for (shape[1..], layers) |dim, *layer| {
            const from = offset;
            const to = offset + dim;
            layer.* = network.nodes[from..to];
            offset = to;
        }

        // Zero initialize.
        for (network.nodes) |*node| {
            node.gradient = [_]f32{0} ** 16;
            //node.weights = [_]f32{0} ** 16;
            node.weights = [_]f32{
                0,
                0,
                0,
                8,
                0,
                8,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            };
            //for (&node.weights) |*w| w.* = rand.float(f32);
            node.children = &.{};
        }

        // Initialize nodes with random parents such that each node has at least one child, excluding the output layer.
        var stack = try ArrayList(usize).initCapacity(allocator, shape[0]);
        defer stack.deinit();
        for (0..shape[0]) |i| stack.appendAssumeCapacity(i);
        rand.shuffle(usize, stack.items);
        for (layers[0]) |*node| {
            node.parents[0] = if (stack.items.len > 0)
                @truncate(stack.pop().?)
            else
                rand.uintLessThan(NodeIndex, @truncate(shape[0]));
            node.parents[1] = if (stack.items.len > 0)
                @truncate(stack.pop().?)
            else while (true) {
                const i = rand.uintLessThan(NodeIndex, @truncate(shape[0]));
                // Fixme: Will fail if shape[0] == 1.
                if (i != node.parents[0]) break i;
            } else unreachable;
        }

        offset = 0;
        for (layers[1..], layers[0 .. layers.len - 1]) |layer, prev| {
            for (0..prev.len) |i| try stack.append(i);
            rand.shuffle(usize, stack.items);
            for (layer) |*node| {
                node.parents[0] = if (stack.items.len > 0)
                    @truncate(offset + stack.pop().?)
                else
                    @truncate(offset + rand.uintLessThan(NodeIndex, @truncate(prev.len)));
                node.parents[1] = if (stack.items.len > 0)
                    @truncate(offset + stack.pop().?)
                else while (true) {
                    const i: NodeIndex = @truncate(offset + rand.uintLessThan(NodeIndex, @truncate(prev.len)));
                    // Fixme: Will fail if shape[0] == 1.
                    if (i != node.parents[0]) break i;
                } else unreachable;
            }
            offset += prev.len;
        }

        // Find the children of each node.
        if (shape.len == 2) return network;

        var buffer = ArrayList(NodeIndex).init(allocator);
        errdefer buffer.deinit();
        errdefer for (network.nodes) |node| allocator.free(node.children);
        offset = 0;
        for (layers[0 .. layers.len - 1], layers[1..]) |layer, next| {
            for (layer, 0..) |*parent, j| {
                const parent_index: NodeIndex = @truncate(offset + j);
                for (next, 0..) |child, k| {
                    const child_index: NodeIndex = @truncate(offset + layer.len + k);
                    if (child.parents[0] == parent_index or
                        child.parents[1] == parent_index) try buffer.append(child_index);
                }
                parent.children = try buffer.toOwnedSlice();
            }
            offset += layer.len;
        }

        return network;
    }
};
