const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const aesia = @import("aesia");

const encoding_bits = 1;
const ImageMNIST = [encoding_bits][28 * 28]f32;
const LabelMNIST = u8;

// "Error function" used belof for the Gaussian CDF.
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

// Used to Z-order input indices.
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

fn loadImages(allocator: Allocator, path: []const u8) ![]ImageMNIST {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var decompressor = std.compress.gzip.decompressor(file.reader());
    var buffered = std.io.bufferedReader(decompressor.reader());
    var reader = buffered.reader();

    // Assert magic value.
    assert(try reader.readByte() == 0); // Specified by IDX
    assert(try reader.readByte() == 0); // --/--
    assert(try reader.readByte() == 0x08); // 0x08 means bytes.
    assert(try reader.readByte() == 0x03); // Three dimensios, image count, rows, and columns.
    const image_count = try reader.readInt(u32, .big);
    std.debug.print("Reading {d} images...\n", .{image_count});
    const row_count = try reader.readInt(u32, .big);
    const col_count = try reader.readInt(u32, .big);
    assert(row_count == 28);
    assert(col_count == 28);

    // Read raw image data and normalize.
    const RawImage = [28 * 28]f32;
    const scale = 1.0 / 255.0;
    const raw_images = try allocator.alloc(RawImage, image_count);
    defer allocator.free(raw_images);
    for (raw_images) |*image| {
        for (image) |*pixel| {
            pixel.* = @as(f32, @floatFromInt(try reader.readByte())) * scale;
        }
    }

    // Standardize using the mean and deviation and thermometer encode.
    // Fashion MNIST:
    // const mean = 0.2994749;
    // const deviation = 0.3270475;
    // Regular MNIST:
    const mean = 0.13113596;
    const deviation = 0.2885157;
    const images = try allocator.alloc(ImageMNIST, image_count);
    for (images, raw_images) |*image, raw_image| {
        inline for (0..encoding_bits) |bit| {
            for (0..28) |row| {
                for (0..28) |col| {
                    const raw_pixel = raw_image[28 * row + col];
                    const z_score = (raw_pixel - mean) / deviation;
                    const probability = zScoreToProbability(z_score);
                    const percentile = @as(f32, @floatFromInt(bit + 1)) /
                        @as(f32, @floatFromInt(encoding_bits + 1));
                    image[bit][28 * row + col] = if (probability >= percentile) 1 else 0;
                }
            }
        }
    }
    return images;
}

fn loadLabels(allocator: Allocator, path: []const u8) ![]LabelMNIST {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var decompressor = std.compress.gzip.decompressor(file.reader());
    var buffered = std.io.bufferedReader(decompressor.reader());
    var reader = buffered.reader();

    // Assert magic value.
    if (try reader.readByte() != 0) return error.InvalidIDXMagicValue;
    if (try reader.readByte() != 0) return error.InvalidIDXMagicValue;
    if (try reader.readByte() != 0x08) return error.InvalidEntrySize;
    if (try reader.readByte() != 0x01) return error.UnexpectedDimensionCount;
    const label_count = try reader.readInt(u32, .big);
    std.debug.print("Reading {d} labels...\n", .{label_count});
    const labels = try allocator.alloc(LabelMNIST, label_count);
    assert(try reader.readAll(labels) == label_count);
    return labels;
}

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();

// Used to pad a 28x28 MNIST image to 32x32 in a random fashion. Essentially it randomly
// translates the input image up or down, left or right. This greatly improves the accuracy
// for bigger models.
const RandomPadding = struct {
    const PaddingLayers = blk: {
        var result: []const type = &.{};
        for (0..3) |row| {
            for (0..3) |col| {
                const left = 3 - col;
                const right = 4 - left;
                const top = 3 - row;
                const bottom = 4 - top;
                result = result ++ .{layer.Pad(.{
                    .depth_in = encoding_bits,
                    .height_in = 28,
                    .width_in = 28,
                    .padding = .{ .top = top, .bottom = bottom, .left = left, .right = right },
                })};
            }
        }
        break :blk result;
    };

    pub const info = PaddingLayers[0].info;

    pub fn eval(input: *const [28][28][encoding_bits]f32, output: *[32][32][encoding_bits]f32) void {
        const index = rand.uintLessThan(usize, PaddingLayers.len);
        inline for (0..PaddingLayers.len) |i| {
            if (index == i) PaddingLayers[i].eval(input, output);
        }
    }

    pub fn validationEval(input: *const [28][28][encoding_bits]f32, output: *[32][32][encoding_bits]f32) void {
        PaddingLayers[4].eval(input, output);
    }

    pub fn forwardPass(input: *const [28][28][encoding_bits]f32, output: *[32][32][encoding_bits]f32) void {
        eval(input, output);
    }
};

const layer = aesia.layer;
const Model = aesia.Model(
    &.{
        RandomPadding,
        layer.Repeat(encoding_bits * 32 * 32, 1 << 13),
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
        layer.GroupSum(1 << 13, 10, 18),
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

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const images_training = try loadImages(allocator, "data/mnist/train-images-idx3-ubyte.gz");
    const labels_training = try loadLabels(allocator, "data/mnist/train-labels-idx1-ubyte.gz");
    const images_validate = try loadImages(allocator, "data/mnist/t10k-images-idx3-ubyte.gz");
    const labels_validate = try loadLabels(allocator, "data/mnist/t10k-labels-idx1-ubyte.gz");

    // Load prior model.
    // std.debug.print("Loading latest mnist.model...", .{});
    // try model.readFromFile("mnist.model");
    // model.lock();
    // std.debug.print(
    //     "training accuracy: {d}\n",
    //     .{model.validationCost(.init(@as([]const [encoding_bits * 3072]f32, @ptrCast(images_training)), labels_training))},
    // );
    // model.unlock();

    const epoch_count = 100;
    const batch_size = 16;
    model.train(
        .init(@as([]const [encoding_bits * 28 * 28]f32, @ptrCast(images_training)), labels_training),
        .init(@as([]const [encoding_bits * 28 * 28]f32, @ptrCast(images_validate)), labels_validate),
        epoch_count,
        batch_size,
    );

    std.debug.print("Writing model to mnist.model\n", .{});
    try model.writeToFile("mnist.model");
}
