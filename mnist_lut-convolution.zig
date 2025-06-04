const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const aesia = @import("aesia.zig");

const dim = 28;
const Image = [dim * dim]f32;
const Label = u8;

fn loadImages(allocator: Allocator, path: []const u8) ![]Image {
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
    assert(row_count == dim);
    assert(col_count == dim);
    const images = try allocator.alloc(Image, image_count);
    const scale = 1.0 / 255.0;
    for (images) |*image| {
        for (image) |*pixel| {
            pixel.* = @as(f32, @floatFromInt(try reader.readByte())) * scale;
        }
    }
    return images;
}

fn loadLabels(allocator: Allocator, path: []const u8) ![]Label {
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
    const labels = try allocator.alloc(Label, label_count);
    assert(try reader.readAll(labels) == label_count);
    return labels;
}

const ConvolutionLogic = aesia.layer.ConvolutionLogic;
const MultiLogicGate = aesia.layer.MultiLogicGate;
const MultiLogicMax = aesia.layer.MultiLogicMax;
const MultiLogicXOR = aesia.layer.MultiLogicXOR;
const LogicLayer = aesia.layer.PackedLogic;
const GroupSum = aesia.layer.GroupSum;
const MaxPool = aesia.layer.MaxPool;
const LUTConvolution = aesia.layer.LUTConvolutionPlies;
const ButterflyMap = @import("dyadic_butterfly.zig").ButterflyMap;

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();
const width = 32_000;
const model_scale = 4;
const Model = aesia.Model(&.{
    LUTConvolution(.{
        .depth = 1,
        .height = 28,
        .width = 28,
        .lut_count = model_scale,
        .field_size = .{ .height = 3, .width = 3 },
        .stride = .{ .row = 1, .col = 1 },
    }),
    LUTConvolution(.{
        .depth = model_scale,
        .height = 26,
        .width = 26,
        .lut_count = 1,
        .field_size = .{ .height = 2, .width = 2 },
        .stride = .{ .row = 2, .col = 2 },
    }),
    LUTConvolution(.{
        .depth = model_scale,
        .height = 13,
        .width = 13,
        .lut_count = 4,
        .field_size = .{ .height = 3, .width = 3 },
        .stride = .{ .row = 2, .col = 2 },
    }),
    LUTConvolution(.{
        .depth = 4 * model_scale,
        .height = 6,
        .width = 6,
        .lut_count = 4,
        .field_size = .{ .height = 2, .width = 2 },
        .stride = .{ .row = 2, .col = 2 },
    }),
    LogicLayer(16 * model_scale * 3 * 3, 32_000, .{ .rand = &rand }),
    LogicLayer(32_000, 16_000, .{ .rand = &rand }),
    LogicLayer(16_000, 8_000, .{ .rand = &rand }),
    GroupSum(8000, 10),
}, .{
    .Loss = aesia.loss.DiscreteCrossEntropy(u8, 10),
    .Optimizer = aesia.optimizer.AdamW(.{
        .learn_rate = 0.05,
        .weight_decay = 0.002,
    }),
});
var model: Model = undefined;

pub fn main() !void {
    model.init();

    const allocator = std.heap.page_allocator;
    const images_training = try loadImages(allocator, "data/train-images-idx3-ubyte.gz");
    const labels_training = try loadLabels(allocator, "data/train-labels-idx1-ubyte.gz");
    const images_validate = try loadImages(allocator, "data/t10k-images-idx3-ubyte.gz");
    const labels_validate = try loadLabels(allocator, "data/t10k-labels-idx1-ubyte.gz");

    // Load prior model.
    // std.debug.print("Loading latest lut-convolution.model...", .{});
    // try model.readFromFile("lut-convolution.model");
    // model.lock();
    // std.debug.print("successfully loaded model with validiation cost: {d}\n", .{model.cost(.init(images_validate, labels_validate))});
    // model.unlock();

    const training_count = 60_000;
    const validate_count = 10_000;

    model.lock();
    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = model.eval(&image);
        if (std.mem.indexOfMax(f32, prediction) == label) correct_count += 1;
    }
    model.unlock();

    std.debug.print(
        "Correctly classified {d} / {d} ~ {d}%\n",
        .{
            correct_count,
            images_validate.len,
            100 * @as(f32, @floatFromInt(correct_count)) /
                @as(f32, @floatFromInt(images_validate.len)),
        },
    );

    var timer = try std.time.Timer.start();
    const epoch_count = 1;
    const batch_size = 32;
    model.train(
        .init(images_training[0..training_count], labels_training[0..training_count]),
        .init(images_validate[0..validate_count], labels_validate[0..validate_count]),
        epoch_count,
        batch_size,
    );

    model.lock();
    correct_count = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = model.eval(&image);
        if (std.mem.indexOfMax(f32, prediction) == label) correct_count += 1;
    }
    model.unlock();

    std.debug.print(
        "Correctly classified {d} / {d} ~ {d}%\n",
        .{
            correct_count,
            images_validate.len,
            100 * @as(f32, @floatFromInt(correct_count)) /
                @as(f32, @floatFromInt(images_validate.len)),
        },
    );
    std.debug.print("Training took: {d}min\n", .{timer.read() / std.time.ns_per_min});

    std.debug.print("Writing model to lut-convolution.model\n", .{});
    try model.writeToFile("lut-convolution.model");
}
