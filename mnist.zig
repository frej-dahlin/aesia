const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const dlg = @import("dlg.zig");

const dim = 28;
const Image = [dim * dim]f32;
const Label = u8;

fn loadImages(allocator: Allocator, path: []const u8) ![]Image {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var buffered = std.io.bufferedReader(file.reader());
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

    var buffered = std.io.bufferedReader(file.reader());
    var reader = buffered.reader();

    // Assert magic value.
    assert(try reader.readByte() == 0); // Specified by IDX
    assert(try reader.readByte() == 0); // --/--
    assert(try reader.readByte() == 0x08); // 0x08 means bytes.
    assert(try reader.readByte() == 0x01); // One dimension, label.
    const label_count = try reader.readInt(u32, .big);
    std.debug.print("Reading {d} labels...\n", .{label_count});
    const labels = try allocator.alloc(Label, label_count);
    assert(try reader.readAll(labels) == label_count);
    return labels;
}

const Logic = dlg.Logic;
const GroupSum = dlg.GroupSum;

// zig fmt: off
const Model = dlg.Model(&.{ 
      Logic(.{ .input_dim = 784,    .output_dim = 16_000, .seed = 0 }), 
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 1 }), 
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 2 }), 
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 3 }), 
      Logic(.{ .input_dim = 16_000, .output_dim = 16_000, .seed = 4 }), 
      GroupSum(16_000, 10), 
}, .{
    .Loss = dlg.loss_function.DiscreteCrossEntropy(u8, 10),
    .Optimizer = dlg.optim.Adam(.{.learn_rate = 0.02}),
}); 
// zig fmt: on
var model: Model = undefined;

pub fn main() !void {
    model.init();

    const allocator = std.heap.page_allocator;
    const images_training = try loadImages(allocator, "data/train-images-idx3-ubyte");
    const labels_training = try loadLabels(allocator, "data/train-labels-idx1-ubyte");
    const images_validate = try loadImages(allocator, "data/t10k-images-idx3-ubyte");
    const labels_validate = try loadLabels(allocator, "data/t10k-labels-idx1-ubyte");

    // Load prior model.
    // It must have been initialized with seed = 0.
    //std.debug.print("Loading latest mnist.model...", .{});
    //const gigabyte = 1_000_000_000;
    //const parameter_bytes = try std.fs.cwd().readFileAlloc(allocator, "mnist.model", gigabyte);
    //defer allocator.free(parameter_bytes);
    //@memcpy(&model.parameters, std.mem.bytesAsSlice(f32, parameter_bytes));
    //model.network.setLogits(@ptrCast(&model.parameters));
    //std.debug.print("successfully loaded model with validiation cost: {d}\n", .{model.cost(.init(images_validate, labels_validate))});

    const training_count = 10_000;
    const validate_count = 10_000;

    var timer = try std.time.Timer.start();
    model.train(.init(images_training[0..training_count], labels_training[0..training_count]), .init(images_validate[0..validate_count], labels_validate[0..validate_count]), 5, 32);
    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        var max_index: usize = 0;
        var max: f32 = 0;
        for (model.eval(&image), 0..) |probability, k| {
            if (probability > max) {
                max = probability;
                max_index = k;
            }
        }
        if (max_index == label) correct_count += 1;
    }
    std.debug.print("Correctly classified {d} / {d} ~ {d}%\n", .{ correct_count, images_validate.len, 100 * @as(f32, @floatFromInt(correct_count)) / @as(f32, @floatFromInt(images_validate.len)) });
    std.debug.print("Training took: {d}min\n", .{timer.read() / std.time.ns_per_min});

    std.debug.print("Writing model to mnist.model\n", .{});
    const file = try std.fs.cwd().createFile("mnist.model", .{});
    defer file.close();
    var buffered = std.io.bufferedWriter(file.writer());
    var writer = buffered.writer();

    try writer.writeAll(std.mem.asBytes(&model.parameters));
}
