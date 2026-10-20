#ifndef include_decompress_image
#define include_decompress_image

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

class DecompressImage {
public:
	// Master image decoder
	static DecodedImage decode_image(const std::filesystem::path& file_path);

	// Individual format decoders
	static DecodedImage decompress_png(const std::filesystem::path& file_path);
	static DecodedImage decompress_jpeg(const std::filesystem::path& file_path);
	static DecodedImage decompress_bmp(const std::filesystem::path& file_path);
	static DecodedImage decompress_webp(const std::filesystem::path& file_path);
};

#endif // include_decompress_image