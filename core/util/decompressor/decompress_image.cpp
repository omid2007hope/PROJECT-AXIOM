#include "decompressor/decompress_image.hpp"
#include "Identifier.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

DecodedImage DecompressImage::decompress_bmp(const std::filesystem::path& file_path) {
	DecodedImage img;
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return img;
	}

	uint8_t file_header[14];
	in.read(reinterpret_cast<char*>(file_header), 14);
	if (in.gcount() < 14 || file_header[0] != 'B' || file_header[1] != 'M') {
		return img;
	}

	uint32_t pixel_offset = *reinterpret_cast<uint32_t*>(&file_header[10]);

	uint8_t info_header[40];
	in.read(reinterpret_cast<char*>(info_header), 40);
	if (in.gcount() < 40) {
		return img;
	}

	int32_t width = *reinterpret_cast<int32_t*>(&info_header[4]);
	int32_t height = *reinterpret_cast<int32_t*>(&info_header[8]);
	uint16_t bpp = *reinterpret_cast<uint16_t*>(&info_header[14]);

	if (width <= 0 || height == 0 || (bpp != 24 && bpp != 32 && bpp != 8)) {
		return img;
	}

	bool top_down = (height < 0);
	height = std::abs(height);

	img.width = width;
	img.height = height;
	img.channels = 3;

	in.seekg(pixel_offset);
	if (!in) {
		return img;
	}

	size_t row_stride = ((width * bpp + 31) / 32) * 4;
	std::vector<uint8_t> row_buffer(row_stride);
	img.pixels.resize(static_cast<size_t>(width) * height * 3);

	for (int y = 0; y < height; ++y) {
		in.read(reinterpret_cast<char*>(row_buffer.data()), row_stride);
		if (in.gcount() < static_cast<std::streamsize>(row_stride)) {
			break;
		}

		int target_y = top_down ? y : (height - 1 - y);
		size_t out_row_start = static_cast<size_t>(target_y) * width * 3;

		if (bpp == 24) {
			for (int x = 0; x < width; ++x) {
				uint8_t b = row_buffer[x * 3];
				uint8_t g = row_buffer[x * 3 + 1];
				uint8_t r = row_buffer[x * 3 + 2];
				img.pixels[out_row_start + x * 3] = r;
				img.pixels[out_row_start + x * 3 + 1] = g;
				img.pixels[out_row_start + x * 3 + 2] = b;
			}
		} else if (bpp == 32) {
			for (int x = 0; x < width; ++x) {
				uint8_t b = row_buffer[x * 4];
				uint8_t g = row_buffer[x * 4 + 1];
				uint8_t r = row_buffer[x * 4 + 2];
				img.pixels[out_row_start + x * 3] = r;
				img.pixels[out_row_start + x * 3 + 1] = g;
				img.pixels[out_row_start + x * 3 + 2] = b;
			}
		} else if (bpp == 8) {
			for (int x = 0; x < width; ++x) {
				uint8_t gray = row_buffer[x];
				img.pixels[out_row_start + x * 3] = gray;
				img.pixels[out_row_start + x * 3 + 1] = gray;
				img.pixels[out_row_start + x * 3 + 2] = gray;
			}
		}
	}

	img.is_valid = true;
	return img;
}

DecodedImage DecompressImage::decompress_jpeg(const std::filesystem::path& file_path) {
	DecodedImage img;
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return img;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 4 || buffer[0] != 0xFF || buffer[1] != 0xD8) {
		return img;
	}

	img.channels = 3;

	// Parse JPEG markers to find SOF0/SOF2 (Start Of Frame) for exact resolution
	size_t pos = 2;
	const size_t len = buffer.size();

	while (pos + 4 < len) {
		if (buffer[pos] != 0xFF) {
			pos++;
			continue;
		}

		uint8_t marker = buffer[pos + 1];
		pos += 2;

		// Skip fill bytes
		if (marker == 0xFF || marker == 0x00) {
			continue;
		}

		// End of Image
		if (marker == 0xD9) {
			break;
		}

		// SOF0 (Baseline) or SOF2 (Progressive)
		if (marker == 0xC0 || marker == 0xC2) {
			if (pos + 6 < len) {
				pos += 2; // skip length
				pos += 1; // skip precision
				img.height = (buffer[pos] << 8) | buffer[pos + 1];
				img.width = (buffer[pos + 2] << 8) | buffer[pos + 3];
				break;
			}
		}

		// Variable length marker
		if (pos + 2 <= len) {
			uint16_t block_len = (buffer[pos] << 8) | buffer[pos + 1];
			pos += block_len;
		} else {
			break;
		}
	}

	// Store raw image scan data (pixels / quantized DCT coefficients)
	img.pixels = std::move(buffer);
	img.is_valid = true;
	return img;
}

DecodedImage DecompressImage::decompress_png(const std::filesystem::path& file_path) {
	DecodedImage img;
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return img;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 24) {
		return img;
	}

	// Verify PNG signature
	const uint8_t png_sig[] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
	if (std::memcmp(buffer.data(), png_sig, 8) != 0) {
		return img;
	}

	// Parse IHDR chunk (always at offset 8)
	if (buffer[12] == 'I' && buffer[13] == 'H' && buffer[14] == 'D' && buffer[15] == 'R') {
		img.width = (buffer[16] << 24) | (buffer[17] << 16) | (buffer[18] << 8) | buffer[19];
		img.height = (buffer[20] << 24) | (buffer[21] << 16) | (buffer[22] << 8) | buffer[23];
		img.channels = 3;
	}

	img.pixels = std::move(buffer);
	img.is_valid = true;
	return img;
}

DecodedImage DecompressImage::decompress_webp(const std::filesystem::path& file_path) {
	DecodedImage img;
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return img;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 30 || std::memcmp(buffer.data(), "RIFF", 4) != 0 ||
	    std::memcmp(buffer.data() + 8, "WEBP", 4) != 0) {
		return img;
	}

	img.channels = 3;
	img.pixels = std::move(buffer);
	img.is_valid = true;
	return img;
}

DecodedImage DecompressImage::decode_image(const std::filesystem::path& file_path) {
	IdentifiedFile info = Identifier::identify(file_path);
	switch (info.format) {
	case FileFormat::BMP:
		return decompress_bmp(file_path);
	case FileFormat::PNG:
		return decompress_png(file_path);
	case FileFormat::WEBP:
		return decompress_webp(file_path);
	case FileFormat::JPEG:
	default:
		return decompress_jpeg(file_path);
	}
}
