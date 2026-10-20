#ifndef include_decompress_video
#define include_decompress_video

#include <cstdint>
#include <filesystem>
#include <vector>

struct DecodedVideo {
	int width = 0;
	int height = 0;
	int frame_count = 0;
	double fps = 0.0;
	std::vector<uint8_t> raw_frames {}; // Sequential RGB frame pixel bytes
	bool is_valid = false;
};

class DecompressVideo {
public:
	// Master video decoder
	static DecodedVideo decode_video(const std::filesystem::path& file_path);

	// Individual format decoders
	static DecodedVideo decompress_mp4(const std::filesystem::path& file_path);
	static DecodedVideo decompress_mkv(const std::filesystem::path& file_path);
};

#endif // include_decompress_video