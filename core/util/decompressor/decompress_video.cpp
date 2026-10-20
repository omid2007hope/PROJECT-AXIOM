#include "decompressor/decompress_video.hpp"
#include "Identifier.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

DecodedVideo DecompressVideo::decompress_mp4(const std::filesystem::path& file_path) {
	DecodedVideo video;
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return video;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 16) {
		return video;
	}

	size_t pos = 0;
	const size_t len = buffer.size();

	// Parse ISO Base Media File Format (MP4 atoms / boxes)
	while (pos + 8 <= len) {
		uint32_t box_size = (buffer[pos] << 24) | (buffer[pos + 1] << 16) |
				    (buffer[pos + 2] << 8) | buffer[pos + 3];
		char box_type[5] = {0};
		std::memcpy(box_type, buffer.data() + pos + 4, 4);

		if (box_size == 0) {
			box_size = static_cast<uint32_t>(len - pos);
		} else if (box_size == 1 && pos + 16 <= len) {
			// Extended 64-bit size
			uint64_t large_size = 0;
			for (int b = 0; b < 8; ++b) {
				large_size = (large_size << 8) | buffer[pos + 8 + b];
			}
			box_size = static_cast<uint32_t>(std::min<uint64_t>(large_size, len - pos));
		}

		if (pos + box_size > len) {
			box_size = static_cast<uint32_t>(len - pos);
		}

		// Locate Track Header Box ('tkhd') inside 'moov' for width and height
		if (std::memcmp(box_type, "moov", 4) == 0) {
			size_t sub = pos + 8;
			size_t moov_end = pos + box_size;
			while (sub + 8 < moov_end) {
				uint32_t sub_size = (buffer[sub] << 24) | (buffer[sub + 1] << 16) |
						    (buffer[sub + 2] << 8) | buffer[sub + 3];
				if (sub_size < 8 || sub + sub_size > moov_end) break;

				if (std::memcmp(buffer.data() + sub + 4, "trak", 4) == 0) {
					// Inside trak, search for tkhd
					size_t trak_sub = sub + 8;
					size_t trak_end = sub + sub_size;
					while (trak_sub + 8 < trak_end) {
						uint32_t t_size = (buffer[trak_sub] << 24) | (buffer[trak_sub + 1] << 16) |
								  (buffer[trak_sub + 2] << 8) | buffer[trak_sub + 3];
						if (t_size < 8 || trak_sub + t_size > trak_end) break;

						if (std::memcmp(buffer.data() + trak_sub + 4, "tkhd", 4) == 0 && t_size >= 84) {
							// In tkhd (v0), width and height are fixed-point 16.16 at offsets 76 and 80
							uint32_t w_fixed = (buffer[trak_sub + 80] << 8) | buffer[trak_sub + 81];
							uint32_t h_fixed = (buffer[trak_sub + 84] << 8) | buffer[trak_sub + 85];
							if (w_fixed > 0 && h_fixed > 0) {
								video.width = static_cast<int>(w_fixed);
								video.height = static_cast<int>(h_fixed);
							}
						}
						trak_sub += t_size;
					}
				}
				sub += sub_size;
			}
		}

		// Locate Media Data Box ('mdat') containing sequential video frames
		if (std::memcmp(box_type, "mdat", 4) == 0 && box_size > 8) {
			size_t mdat_payload_size = box_size - 8;
			video.raw_frames.insert(video.raw_frames.end(),
						buffer.begin() + pos + 8,
						buffer.begin() + pos + 8 + mdat_payload_size);
		}

		pos += box_size;
	}

	if (video.width == 0) video.width = 1920;
	if (video.height == 0) video.height = 1080;
	video.fps = 30.0;
	video.frame_count = static_cast<int>(video.raw_frames.size() / (video.width * video.height * 3 + 1));
	if (video.frame_count == 0 && !video.raw_frames.empty()) video.frame_count = 1;

	video.is_valid = !video.raw_frames.empty();
	return video;
}

DecodedVideo DecompressVideo::decompress_mkv(const std::filesystem::path& file_path) {
	DecodedVideo video;
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return video;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 16) {
		return video;
	}

	video.width = 1920;
	video.height = 1080;
	video.fps = 30.0;
	video.raw_frames = std::move(buffer);
	video.frame_count = 1;
	video.is_valid = !video.raw_frames.empty();

	return video;
}

DecodedVideo DecompressVideo::decode_video(const std::filesystem::path& file_path) {
	IdentifiedFile info = Identifier::identify(file_path);
	switch (info.format) {
	case FileFormat::MKV:
	case FileFormat::WEBM:
		return decompress_mkv(file_path);
	case FileFormat::MP4:
	default:
		return decompress_mp4(file_path);
	}
}
