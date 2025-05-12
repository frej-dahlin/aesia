const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const skiffer = @import("skiffer.zig");

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

const LogicLayer = skiffer.layer.PackedLogic;
const GroupSum = skiffer.layer.GroupSum;

const width = 8_000;
const Model = skiffer.Model(&.{
    LogicLayer(784, width, .{ .seed = 0 }),
    LogicLayer(width, width, .{ .seed = 1 }),
    LogicLayer(width, width, .{ .seed = 2 }),
    LogicLayer(width, width, .{ .seed = 3 }),
    LogicLayer(width, width, .{ .seed = 4 }),
    LogicLayer(width, width, .{ .seed = 5 }),
    GroupSum(width, 10),
}, .{
    .Loss = skiffer.loss.DiscreteCrossEntropy(u8, 10),
    .Optimizer = skiffer.optimizer.Adam(.{ .learn_rate = 0.02 }),
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
    // It must have been initialized with seed = 0.
    // std.debug.print("Loading latest mnist.model...", .{});
    // const gigabyte = 1_000_000_000;
    // const parameter_bytes = try std.fs.cwd().readFileAlloc(allocator, "mnist.model", gigabyte);
    // defer allocator.free(parameter_bytes);
    // @memcpy(&model.parameters, std.mem.bytesAsSlice(f32, parameter_bytes));
    // model.lock();
    // std.debug.print("successfully loaded model with validiation cost: {d}\n", .{model.cost(.init(images_validate, labels_validate))});
    // model.unlock();

    const training_count = 60_000;
    const validate_count = 10_000;

    var timer = try std.time.Timer.start();
    const epoch_count = 30;
    const batch_size = 32;
    model.train(.init(images_training[0..training_count], labels_training[0..training_count]), .init(images_validate[0..validate_count], labels_validate[0..validate_count]), epoch_count, batch_size);

    model.lock();
    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = model.eval(&image);
        if (std.mem.indexOfMax(f32, prediction) == label) correct_count += 1;
    }
    model.unlock();
    std.debug.print("Correctly classified {d} / {d} ~ {d}%\n", .{ correct_count, images_validate.len, 100 * @as(f32, @floatFromInt(correct_count)) / @as(f32, @floatFromInt(images_validate.len)) });
    std.debug.print("Training took: {d}min\n", .{timer.read() / std.time.ns_per_min});

    std.debug.print("Writing model to mnist.model\n", .{});
    const file = try std.fs.cwd().createFile("mnist.model", .{});
    defer file.close();
    var buffered = std.io.bufferedWriter(file.writer());
    var writer = buffered.writer();

    try writer.writeAll(std.mem.asBytes(&model.parameters));
}
