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
            pixel.* = if (@as(f32, @floatFromInt(try reader.readByte())) * scale > 0.5) 1 else 0;
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

const layer = aesia.layer;

const Model = aesia.Model(&.{
    layer.Repeat(784, 8192),
    layer.ButterflyMap(13, 12),
    layer.ButterflyMap(13, 11),
    layer.ButterflyMap(13, 10),
    layer.ButterflyMap(13, 9),
    layer.ButterflyMap(13, 8),
    layer.ButterflyMap(13, 7),
    layer.ButterflyMap(13, 6),
    layer.ButterflyMap(13, 5),
    layer.ButterflyMap(13, 4),
    layer.ButterflyMap(13, 3),
    layer.ButterflyMap(13, 2),
    layer.ButterflyMap(13, 1),
    layer.ButterflyMap(13, 0),
    layer.ButterflyMap(13, 1),
    layer.ButterflyMap(13, 2),
    layer.ButterflyMap(13, 3),
    layer.ButterflyMap(13, 4),
    layer.ButterflyMap(13, 5),
    layer.ButterflyMap(13, 6),
    layer.ButterflyMap(13, 7),
    layer.ButterflyMap(13, 8),
    layer.ButterflyMap(13, 9),
    layer.ButterflyMap(13, 10),
    layer.ButterflyMap(13, 11),
    layer.ButterflyMap(13, 12),
    layer.LogicSequential(4096),
    layer.Repeat(4096, 8192),
    layer.ButterflyMap(13, 12),
    layer.ButterflyMap(13, 11),
    layer.ButterflyMap(13, 10),
    layer.ButterflyMap(13, 9),
    layer.ButterflyMap(13, 8),
    layer.ButterflyMap(13, 7),
    layer.ButterflyMap(13, 6),
    layer.ButterflyMap(13, 5),
    layer.ButterflyMap(13, 4),
    layer.ButterflyMap(13, 3),
    layer.ButterflyMap(13, 2),
    layer.ButterflyMap(13, 1),
    layer.ButterflyMap(13, 0),
    layer.LogicSequential(4096),
    layer.Repeat(4096, 8192),
    layer.ButterflyMap(13, 0),
    layer.ButterflyMap(13, 1),
    layer.ButterflyMap(13, 2),
    layer.ButterflyMap(13, 3),
    layer.ButterflyMap(13, 4),
    layer.ButterflyMap(13, 5),
    layer.ButterflyMap(13, 6),
    layer.ButterflyMap(13, 7),
    layer.ButterflyMap(13, 8),
    layer.ButterflyMap(13, 9),
    layer.ButterflyMap(13, 10),
    layer.ButterflyMap(13, 11),
    layer.ButterflyMap(13, 12),
    layer.LogicSequential(4096),
    layer.GroupSum(1 << 12, 10, 10.0),
}, .{
    .Loss = aesia.loss.DiscreteCrossEntropy(u8, 10),
    .ValidationLoss = aesia.loss.MissClassificationCount(u8, 10),
    .Optimizer = aesia.optimizer.Adam(.{
        .learn_rate = 0.01,
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
    // std.debug.print("Loading latest mnist.model...", .{});
    // try model.readFromFile("mnist.model");
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
    const epoch_count = 100;
    const batch_size = 16;
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

    std.debug.print("Writing model to mnist.model\n", .{});
    try model.writeToFile("mnist.model");
}
