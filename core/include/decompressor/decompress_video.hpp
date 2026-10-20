#ifndef include_decom_video
#define include_decom_video

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

class Decom_video {
public:
	// Master video decoder
	static DecodedVideo decode_video(const std::filesystem::path& filepath);

	// Individual format decoders
	static DecodedVideo decom_MP4(const std::filesystem::path& filepath);
	static DecodedVideo decom_MKV(const std::filesystem::path& filepath);
};

#endif // include_decom_video