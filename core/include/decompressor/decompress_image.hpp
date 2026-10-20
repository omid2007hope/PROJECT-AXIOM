#ifndef include_decom_image
#define include_decom_image

#include <cstdint>
#include <filesystem>
#include <vector>

struct DecodedImage {
	int width = 0;
	int height = 0;
	int channels = 0;               // 3 = RGB, 1 = Grayscale, 4 = RGBA
	std::vector<uint8_t> pixels {}; // Raw RGB pixel bytes [0x00 .. 0xFF]
	bool is_valid = false;
};

class Decom_image {
public:
	// Master image decoder
	static DecodedImage decode_image(const std::filesystem::path& filepath);

	// Individual format decoders
	static DecodedImage PNG(const std::filesystem::path& filepath);
	static DecodedImage decom_JPG(const std::filesystem::path& filepath);
	static DecodedImage decom_BMP(const std::filesystem::path& filepath);
	static DecodedImage decom_WEBP(const std::filesystem::path& filepath);
};

#endif // include_decom_image