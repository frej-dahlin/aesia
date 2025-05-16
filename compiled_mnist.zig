const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const skiffer = @import("skiffer.zig");

const dim = 28;
const Image = [dim * dim]usize;
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
    for (images) |*image| {
        for (image) |*pixel| {
            pixel.* = if (try .reader.readByte() > 0) 1 else 0;
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

const compiled_layer = @import("compiled_layer.zig");
const LogicLayer = compiled_layer.Logic;
const GroupSum = compiled_layer.GroupSum;

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();
const width = 8000;
const Network = @import("compiled_network.zig").Network(&.{
    LogicLayer(784, width, .{ .rand = &rand }),
    LogicLayer(width, width, .{ .rand = &rand }),
    LogicLayer(width, width, .{ .rand = &rand }),
    LogicLayer(width, width, .{ .rand = &rand }),
    LogicLayer(width, width, .{ .rand = &rand }),
    LogicLayer(width, width, .{ .rand = &rand }),
    GroupSum(width, 10),
});
var model: Model = undefined;

pub fn main() !void {
    model.compileFromFile("mnist.model");

    const allocator = std.heap.page_allocator;
    const images_validate = try loadImages(allocator, "data/t10k-images-idx3-ubyte.gz");
    const labels_validate = try loadLabels(allocator, "data/t10k-labels-idx1-ubyte.gz");

    const training_count = 1_000;
    const validate_count = 10_000;

    var timer = try std.time.Timer.start();

    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = network.eval(&image);
        if (std.mem.indexOfMax(f32, prediction) == label) correct_count += 1;
    }

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
}
