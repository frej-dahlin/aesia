pub const std = @import("std");
const assert = std.debug.assert;

pub const aesia = @import("aesia.zig");

pub const ModelOptions = struct {
    Optimizer: ?fn (usize) type,
    Loss: ?type,

    pub const default = ModelOptions{
        .Optimizer = null,
        .Loss = null,
    };
};

pub fn Model(Layers: []const type, options: ModelOptions) type {
    return struct {
        const Self = @This();

        const Network = aesia.Network(Layers);
        pub const parameter_count = Network.parameter_count;
        const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
        comptime {
            assert(parameter_count % vector_len == 0);
        }
        const parameter_alignment = @max(Network.parameter_alignment, @alignOf(@Vector(vector_len, f32)));

        pub const Feature = Network.Input;
        pub const Prediction = Network.Output;
        pub const input_dim = Network.input_dim;
        pub const output_dim = Network.output_dim;
        pub const Optimizer = (if (options.Optimizer) |O| O else aesia.optimizer.Adam(.default))(parameter_count);
        pub const Loss = if (options.Loss) |L| L else aesia.loss.HalvedMeanSquareError(output_dim);
        pub const Label = Loss.Label;

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

        network: Network,
        parameters: [parameter_count]f32 align(parameter_alignment),
        gradient: [parameter_count]f32 align(parameter_alignment),
        optimizer: Optimizer,
        locked: bool,

        pub fn init(self: *Self) void {
            self.network.init(&self.parameters);
            self.optimizer = .default;
            self.locked = false;
        }

        pub fn eval(model: *Self, input: *const Feature) *const Prediction {
            assert(model.locked);
            return model.network.eval(input);
        }

        pub fn forwardPass(model: *Self, feature: *const Feature) *const Prediction {
            assert(model.locked);
            return model.network.forwardPass(feature);
        }

        pub fn backwardPass(model: *Self) void {
            assert(model.locked);
            return model.network.backwardPass(&model.gradient);
        }

        pub fn lock(model: *Self) void {
            assert(!model.locked);
            model.network.takeParameters(&model.parameters);
            model.locked = true;
        }

        pub fn unlock(model: *Self) void {
            assert(model.locked);
            model.network.giveParameters();
            model.locked = false;
        }

        pub fn loss(model: *Self, feature: *const Feature, label: *const Label) f32 {
            assert(model.locked);
            return Loss.eval(model.eval(feature), label);
        }

        /// Returns the mean loss over a dataset.
        pub fn cost(model: *Self, dataset: Dataset) f32 {
            assert(dataset.len() > 0);
            assert(model.locked);
            var result: f32 = 0;
            for (dataset.features, dataset.labels) |feature, label| result += model.loss(&feature, &label);
            return result / @as(f32, @floatFromInt(dataset.len()));
        }

        pub fn differentiate(model: *Self, dataset: Dataset) f32 {
            @setFloatMode(.optimized);
            assert(model.locked);
            @memset(&model.gradient, 0);
            var total_loss: f32 = 0;
            for (dataset.features, dataset.labels) |feature, label| {
                const prediction = model.forwardPass(&feature);
                Loss.gradient(prediction, &label, model.network.lastDeltaBuffer());
                total_loss += Loss.eval(prediction, &label);
                model.backwardPass();
            }
            return total_loss;
        }

        /// Trains the model on a given dataset for a specified amount of epochs and batch size.
        /// Every epoch the model is 'validated' on another given dataset.
        pub fn train(model: *Self, training: Dataset, validate: Dataset, epoch_count: usize, batch_size: usize) void {
            assert(!model.locked);
            assert(batch_size > 0);
            for (0..epoch_count) |epoch| {
                var timer = std.time.Timer.start() catch unreachable;
                var offset: usize = 0;
                var training_loss: f32 = 0;
                while (training.len() > offset) : (offset += batch_size) {
                    model.lock();
                    const subset = training.slice(offset, @min(offset + batch_size, training.len()));
                    training_loss += model.differentiate(subset);
                    model.unlock();
                    model.optimizer.step(@ptrCast(&model.parameters), @ptrCast(&model.gradient));
                }
                model.lock();
                training_loss /= @as(f32, @floatFromInt(training.len()));
                if (validate.len() > 0) std.debug.print(
                    "EPOCH {d}\ntraining loss: {d:2.4}\nvalidate loss: {d:2.4}\nelapsed time: {d}s\n\n",
                    .{
                        epoch,
                        training_loss,
                        model.cost(validate),
                        timer.read() / std.time.ns_per_s,
                    },
                );
                model.unlock();
            }
        }

        pub fn writeToFile(self: *Self, path: []const u8) !void {
            try Network.writeToFile(&self.parameters, path);
        }

        pub fn readFromFile(self: *Self, path: []const u8) !void {
            try Network.readFromFile(&self.parameters, path);
        }
    };
}
