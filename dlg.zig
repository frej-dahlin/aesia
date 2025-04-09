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
    @setFloatMode(.optimized);
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

    nodes: MultiArrayList(Node),
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

        @setFloatMode(.optimized);
        const nodes = network.nodes.slice();
        const shape = network.shape;

        const to = shape[1];
        for (nodes.items(.value)[0..to],
             nodes.items(.parents)[0..to],
             nodes.items(.weights)[0..to]) |*value, parents, weights| {
            const a = x[parents[0]];
            const b = x[parents[1]];
            value.* = Node.eval_(weights, a, b); 
        }
        const from = to;
        for (nodes.items(.value)[from..],
             nodes.items(.parents)[from..],
             nodes.items(.weights)[from..]) |*value, parents, weights| {
            const a = nodes.items(.value)[parents[0]];
            const b = nodes.items(.value)[parents[1]];
            value.* = Node.eval_(weights, a, b); 
        }
    }

    // Compute the delta for each node relative to a given datapoint.
    // See the definition of the Node struct.
    // Fixme: Currently assumes the network is trained with a halved mean square error.
    pub fn backprop(network: Network, x: []const f32, y: []const f32) void {
        @setFloatMode(.optimized);
        assert(x.len == network.shape[0]);
        assert(y.len == network.shape[network.shape.len - 1]);

        network.eval(x);

        const nodes = network.nodes.slice();
        const shape = network.shape;

        // Compute the delta of the last layer.
        var offset = nodes.len;
        {
            const from = offset - shape[shape.len - 1];	
            const to = offset;
            for (nodes.items(.delta)[from..to], nodes.items(.value)[from..to], y) |*delta, value, y_val| {
                // Fixme: Hardcoded gradient of the halved mean square error.
                delta.* = value - y_val;
            }
            offset = from;
            // Fallthrough!
        }
        // Compute the delta of the previous layers, back to front.
        for (2..shape.len) |i| {
            const from = offset - shape[shape.len - i];
            const to = offset;
            for (nodes.items(.delta)[from..to],
                 nodes.items(.children)[from..to],
                 from..to) |*delta, children, parent_index| {
                delta.* = 0;
                for (children) |child_index| {
                    const child = nodes.get(child_index);
                    const a = nodes.items(.value)[child.parents[0]];
                    const b = nodes.items(.value)[child.parents[1]];
                    delta.* += 1.1 * child.delta *
                        if (child.parents[0] == parent_index) child.del_a(b)
                        else child.del_b(a);
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

        @setFloatMode(.optimized);
        network.backprop(x, y);
        const nodes = network.nodes.slice();

        // Note: Can use inline for-loops to contract these into one.
        // weights, gradient, delta, parents.
        const to = network.shape[1];
        for (nodes.items(.parents)[0..to],
                 nodes.items(.weights)[0..to],
                 nodes.items(.gradient)[0..to],
                 nodes.items(.delta)[0..to]) |parents, weights, *gradient, delta| {
            const a = x[parents[0]];
            const b = x[parents[1]];
            const sigma = softmax(weights);
            const gate = SoftGate.vector(a, b);
            const e1: v16f32 = .{1} ++ .{0} ** 15;
            var result: [16]f32 = .{0} ** 16;
            inline for (&result, 0..) |*partial, k| {
                const kronecker = std.simd.rotateElementsRight(e1, k);
                const sigma_k: v16f32 = @splat(@reduce(.Add, kronecker * sigma));
                partial.* += @reduce(.Add, sigma * gate * (kronecker - sigma_k) * @as(v16f32, @splat(delta)));
            }
            gradient.* = result;
        } 
    
        const from = to;
        for (nodes.items(.parents)[from..],
                 nodes.items(.weights)[from..],
                 nodes.items(.gradient)[from..],
                 nodes.items(.delta)[from..]) |parents, weights, *gradient, delta| {
            const a = nodes.items(.value)[parents[0]];
            const b = nodes.items(.value)[parents[1]];
            const sigma = softmax(weights);
            const gate = SoftGate.vector(a, b);
            const e1: v16f32 = .{1} ++ .{0} ** 15;
            var result: [16]f32 = .{0} ** 16;
            inline for (&result, 0..) |*partial, k| {
                const kronecker = std.simd.rotateElementsRight(e1, k);
                const sigma_k: v16f32 = @splat(@reduce(.Add, kronecker * sigma));
                partial.* += @reduce(.Add, sigma * gate * (kronecker - sigma_k) * @as(v16f32, @splat(delta)));
            }
            gradient.* = result;
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

        network.nodes = .{};
        try network.nodes.resize(allocator, node_count);
        errdefer network.nodes.deinit(allocator);

        // Temporary slice of the layer bounds.
        const Range = struct {from: usize, to: usize};
        const layers: []Range = try allocator.alloc(Range, shape.len - 1);
        defer allocator.free(layers);
        var offset: usize = 0;
        for (shape[1..], layers) |dim, *layer| {
            layer.from = offset;
            layer.to = offset + dim;
            offset += dim;
        }
        
        // Initialize nodes.
        const nodes = network.nodes.slice();
        for (nodes.items(.weights)) |*gradient| {
            // Initialize weights to be biased toward the pass through gates.
            gradient.* = [_]f32{0, 0, 0, 8, 0, 8} ++ .{0} ** 10;
        }
        @memset(nodes.items(.gradient), [_]f32{0} ** 16);
        @memset(nodes.items(.children), &.{});
        
        var stack = try ArrayList(usize).initCapacity(allocator, std.mem.max(usize, shape));	
        defer stack.deinit();
        for (0..shape[0]) |i| stack.appendAssumeCapacity(i);
        rand.shuffle(usize, stack.items);
        for (nodes.items(.parents)[layers[0].from..layers[0].to]) |*parents| {
            const first = stack.pop() orelse rand.uintLessThan(usize, shape[0]);
            const second = stack.pop() orelse while (true) {
                const i = rand.uintLessThan(usize, shape[0]);
                // Fixme: Will fail if shape[0] == 1.
                if (i != first) break i;
            } else unreachable;
            parents.* = .{@truncate(first), @truncate(second)};
        }
        
        for (layers[1..], layers[0..layers.len - 1]) |layer, prev| {
            for (prev.from..prev.to) |i| try stack.append(i);
            rand.shuffle(usize, stack.items);
            for (nodes.items(.parents)[layer.from..layer.to]) |*parents| {
                const first = stack.pop() orelse rand.intRangeLessThan(usize, prev.from, prev.to);
                const second = stack.pop() orelse while (true) {
                    const i = rand.intRangeLessThan(usize, prev.from, prev.to);	
                    // Fixme: Will fail if prev.from == prev.to - 1.
                    if (i != first) break i;
                } else unreachable;
                parents.* = .{@truncate(first), @truncate(second)};
            }
        }

        // Find the children of each node.
        if (shape.len == 2) return network;

        var buffer = ArrayList(NodeIndex).init(allocator);
        errdefer buffer.deinit();
        errdefer for (nodes.items(.children)) |children| allocator.free(children);
        for (layers[0..layers.len - 1], layers[1..]) |layer, next| {
            for (nodes.items(.children)[layer.from..layer.to], layer.from..layer.to) |*children, parent_index| {
                for (nodes.items(.parents)[next.from..next.to], next.from..next.to) |parents, child_index| {
                    if (parents[0] == parent_index or parents[1] == parent_index) try buffer.append(@truncate(child_index));
                }
                children.* = try buffer.toOwnedSlice();
            }
        }
        
        return network;
    }
};
