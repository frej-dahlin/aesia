const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const aesia = @import("aesia");

const encoding_bits = 3;
const CIFAR10Image = [3][encoding_bits][32 * 32]f32;
const CIFAR10Label = u8;

fn erf(x_: f64) f32 {
    // constants
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    // Save the sign of x
    const sign = std.math.sign(x_);
    const x = @abs(x_);

    // A&S formula 7.1.26
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-x * x);

    return @floatCast(sign * y);
}

// The Gaussian cumulative distribution function.
fn zScoreToProbability(z: f32) f32 {
    return 0.5 * (1 + erf(z / @sqrt(2.0)));
}

// Used for locality sensitive Z-ordering.
fn interleaveWithZeros(input: u32) u64 {
    var word: u64 = input;
    word = (word ^ (word << 16)) & 0x0000ffff0000ffff;
    word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f;
    word = (word ^ (word << 2)) & 0x3333333333333333;
    word = (word ^ (word << 1)) & 0x5555555555555555;
    return word;
}

fn interleave(x: u32, y: u32) u64 {
    return (interleaveWithZeros(x) << 1) | interleaveWithZeros(y);
}

fn readCIFAR10Dataset(
    allocator: Allocator,
    filepaths: []const []const u8,
) !std.meta.Tuple(&.{ []const CIFAR10Image, []const CIFAR10Label }) {
    const images_per_file = 10_000;
    const dataset_length = images_per_file * filepaths.len;

    const RawCIFAR10Image = [3][32 * 32]f32;

    var mean_image: RawCIFAR10Image = @splat(@splat(0));
    const raw_images = try allocator.alloc(RawCIFAR10Image, dataset_length);
    defer allocator.free(raw_images);
    const labels = try allocator.alloc(CIFAR10Label, dataset_length);
    for (filepaths, 0..) |filepath, file_index| {
        const file = try std.fs.cwd().openFile(filepath, .{});
        defer file.close();

        // var decompressor = std.compress.gzip.decompressor(file.reader());
        var decompressor = std.compress.gzip.decompressor(file.reader());
        var buffered = std.io.bufferedReader(decompressor.reader());
        var reader = buffered.reader();

        const from = images_per_file * file_index;
        const to = images_per_file * (file_index + 1);

        const normalization = 1.0 / 255.0;
        const mean_scale = 1.0 / @as(f32, @floatFromInt(dataset_length));
        for (labels[from..to], raw_images[from..to]) |*label, *image| {
            label.* = try reader.readByte();
            for (image, &mean_image) |*channel, *mean_channel| {
                for (channel, mean_channel) |*value, *mean| {
                    value.* = @as(f32, @floatFromInt(try reader.readByte())) * normalization;
                    mean.* += mean_scale * value.*;
                }
            }
        }
    }
    const images = try allocator.alloc(CIFAR10Image, dataset_length);
    // Binarize around the means.
    const channel_means: [3]f32 = .{ 0.4914, 0.4822, 0.4465 };
    const channel_deviations: [3]f32 = .{ 0.247, 0.243, 0.261 };
    for (images, raw_images) |*image, *raw_image| {
        inline for (0..3) |channel| {
            inline for (0..encoding_bits) |bit| {
                for (0..32) |row| {
                    for (0..32) |col| {
                        // We use Z-Ordering to preserve locality.
                        const pixel = interleave(@truncate(row), @truncate(col));
                        const percentile = @as(f32, @floatFromInt(bit + 1)) /
                            @as(f32, @floatFromInt(encoding_bits + 1));
                        const raw_value = raw_image[channel][32 * row + col];
                        const z_score = (raw_value - channel_means[channel]) /
                            channel_deviations[channel];
                        const probability = zScoreToProbability(z_score);
                        image[channel][bit][pixel] = if (probability >= percentile) 1 else 0;
                    }
                }
            }
        }
    }

    return .{ images, labels };
}

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();
const RandomCIFAR10Translation = struct {
    pub const info = aesia.layer.Info{
        .dim_in = encoding_bits * 3 * 32 * 32,
        .dim_out = encoding_bits * 3 * 32 * 32,
        .trainable = false,
        .statefull = false,
    };

    var framed_buffer: [34][34]f32 = undefined;

    pub fn eval(input: *const [encoding_bits][3][32][32]f32, output: *[encoding_bits][3][32][32]f32) void {
        const offset_row = rand.intRangeAtMost(isize, -1, 1);
        const offset_col = rand.intRangeAtMost(isize, -1, 1);

        @setEvalBranchQuota(1000 * 32 * 32);
        for (0..encoding_bits) |bit| {
            for (0..3) |channel| {
                inline for (0..32) |row| {
                    inline for (0..32) |col| {
                        framed_buffer[@intCast(1 + offset_row + row)][@intCast(1 + offset_col + col)] =
                            input[bit][channel][row][col];
                    }
                }
                for (0..32) |row| {
                    for (0..32) |col| {
                        output[bit][channel][row][col] = framed_buffer[1 + row][1 + col];
                    }
                }
            }
        }
    }

    pub fn forwardPass(input: *const [encoding_bits][3][32][32]f32, output: *[encoding_bits][3][32][32]f32) void {
        eval(input, output);
    }

    pub fn validationEval(input: *const [encoding_bits][3][32][32]f32, output: *[encoding_bits][3][32][32]f32) void {
        output.* = input.*;
    }
};

const layer = aesia.layer;
const Model = aesia.Model(
    &.{
        // RandomCIFAR10Translation,
        layer.Repeat(encoding_bits * 3 * 32 * 32, 1 << 13),
        layer.ButterflyGate(13, 0),
        layer.ButterflyGate(13, 1),
        layer.ButterflyGate(13, 2),
        layer.ButterflyGate(13, 3),
        layer.ButterflyGate(13, 4),
        layer.ButterflyGate(13, 5),
        layer.ButterflyGate(13, 6),
        layer.ButterflyGate(13, 7),
        layer.ButterflyGate(13, 8),
        layer.ButterflyGate(13, 9),
        layer.ButterflyGate(13, 10),
        layer.ButterflyGate(13, 11),
        layer.ButterflyGate(13, 12),
        layer.GroupSum(1 << 13, 10, 30),
    },
    .{
        .Loss = aesia.loss.DiscreteCrossEntropy(u8, 10),
        .ValidationLoss = aesia.loss.MissClassificationCount(u8, 10),
        .Optimizer = aesia.optimizer.AdamW(.{
            .learn_rate = 0.02,
            .weight_decay = 0.0001,
            .beta = .{
                0.9,
                0.999,
            },
            .schedules = &.{
                aesia.schedule.LinearWarmUp(.{
                    .epoch_count = 2,
                    .learn_rate_max = 0.02,
                    .learn_rate_min = 0.0001,
                }),
                aesia.schedule.LinearWarmDown(.{
                    .field_name = "learn_rate",
                    .max = 0.02,
                    .min = 0.0001,
                    .percent_begin = 0.5,
                }),
            },
        }),
    },
);
var model: Model = undefined;

pub fn main() !void {
    model.init();

    const allocator = std.heap.page_allocator;
    const images_training, const labels_training = try readCIFAR10Dataset(
        allocator,
        &.{
            "data/cifar-10/data_batch_1.bin.gz",
            "data/cifar-10/data_batch_2.bin.gz",
            "data/cifar-10/data_batch_3.bin.gz",
            "data/cifar-10/data_batch_4.bin.gz",
            "data/cifar-10/data_batch_5.bin.gz",
        },
    );
    const images_validate, const labels_validate = try readCIFAR10Dataset(
        allocator,
        &.{
            "data/cifar-10/test_batch.bin.gz",
        },
    );

    const epoch_count = 500;
    const batch_size = 16;
    model.train(
        .init(@as([]const [encoding_bits * 3072]f32, @ptrCast(images_training)), labels_training),
        .init(@as([]const [encoding_bits * 3072]f32, @ptrCast(images_validate)), labels_validate),
        epoch_count,
        batch_size,
    );

    std.debug.print("Writing model to cifar-10.model\n", .{});
    try model.writeToFile("cifar10.model");
}
