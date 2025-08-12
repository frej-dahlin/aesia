const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const rep : compiled_layer.GateRepresentation = compiled_layer.GateRepresentation.boolarray;
const StaticBitSet = @import("bitset.zig").StaticBitSet;

const aesia = @import("aesia.zig");

const dim = 28;
const Image = if(rep == .bitset) StaticBitSet(dim*dim) else [dim * dim]bool;
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
        for (0..dim*dim) |k| {
            if(rep == .bitset){
                image.setValue(k, if (try reader.readByte() > 0) true else false);
            }
            else{
                image[k] = if (try reader.readByte() > 0) true else false;
            }
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
    LogicLayer(784, width, .{ .rand = &rand, .gateRepresentation = rep}),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    GroupSum(width, 10, .{ .rand = &rand, .gateRepresentation = rep }),
});
var network: Network = undefined;

pub fn main() !void {
    try network.compileFromFile("mnist.model");

    const allocator = std.heap.page_allocator;
    const images_validate = try loadImages(allocator, "data/t10k-images-idx3-ubyte.gz");
    const labels_validate = try loadLabels(allocator, "data/t10k-labels-idx1-ubyte.gz");

    var timer = try std.time.Timer.start();

    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = network.eval(&image);
        if (std.mem.indexOfMax(usize, prediction) == label) correct_count += 1;
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
    std.debug.print("Evaluation took: {d}ms\n", .{timer.read() / std.time.ns_per_ms});

    std.debug.print("Permutation took: {d}ms\n", .{network.layers[1].getPermTime() / std.time.ns_per_ms});
    std.debug.print("Gate evaluation took: {d}us\n", .{network.layers[1].getEvalTime() / std.time.ns_per_us});
}
