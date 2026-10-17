#ifndef include_decom_audio
#define include_decom_audio

#include <cstdint>
#include <filesystem>
#include <vector>

struct DecodedAudio {
	int sample_rate = 0;
	int channels = 0;
	int bits_per_sample = 16;
	std::vector<uint8_t> pcm_bytes {}; // Raw linear PCM waveform samples [0x00 .. 0xFF]
	bool is_valid = false;
};

class Decom_audio {
public:
	// Master audio decoder
	static DecodedAudio decode_audio(const std::filesystem::path& filepath);

	// Individual format decoders
	static DecodedAudio decom_WAV(const std::filesystem::path& filepath);
	static DecodedAudio decom_OGG(const std::filesystem::path& filepath);
	static DecodedAudio decom_MP3(const std::filesystem::path& filepath);
};

#endif // include_decom_audio