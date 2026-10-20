#include "decompressor/decom_audio.hpp"
#include "Identifier.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

DecodedAudio Decom_audio::decom_WAV(const std::filesystem::path& filepath) {
	DecodedAudio audio;
	std::ifstream in(filepath, std::ios::binary);
	if (!in) {
		return audio;
	}

	char riff_header[12];
	in.read(riff_header, 12);
	if (in.gcount() < 12 || std::memcmp(riff_header, "RIFF", 4) != 0 ||
	    std::memcmp(riff_header + 8, "WAVE", 4) != 0) {
		return audio;
	}

	bool found_fmt = false;
	bool found_data = false;

	while (in) {
		char chunk_id[4];
		uint32_t chunk_size = 0;
		in.read(chunk_id, 4);
		in.read(reinterpret_cast<char*>(&chunk_size), 4);
		if (in.gcount() < 4) {
			break;
		}

		if (std::memcmp(chunk_id, "fmt ", 4) == 0 && chunk_size >= 16) {
			uint16_t audio_format = 0;
			uint16_t num_channels = 0;
			uint32_t sample_rate = 0;
			uint32_t byte_rate = 0;
			uint16_t block_align = 0;
			uint16_t bits_per_sample = 0;

			in.read(reinterpret_cast<char*>(&audio_format), 2);
			in.read(reinterpret_cast<char*>(&num_channels), 2);
			in.read(reinterpret_cast<char*>(&sample_rate), 4);
			in.read(reinterpret_cast<char*>(&byte_rate), 4);
			in.read(reinterpret_cast<char*>(&block_align), 2);
			in.read(reinterpret_cast<char*>(&bits_per_sample), 2);

			audio.channels = num_channels;
			audio.sample_rate = sample_rate;
			audio.bits_per_sample = bits_per_sample;
			found_fmt = true;

			// Skip extra fmt bytes if chunk_size > 16
			if (chunk_size > 16) {
				in.seekg(chunk_size - 16, std::ios::cur);
			}
		} else if (std::memcmp(chunk_id, "data", 4) == 0) {
			audio.pcm_bytes.resize(chunk_size);
			in.read(reinterpret_cast<char*>(audio.pcm_bytes.data()), chunk_size);
			found_data = true;
			break;
		} else {
			// Skip unrecognized chunk
			in.seekg(chunk_size, std::ios::cur);
		}
	}

	audio.is_valid = (found_fmt && found_data && !audio.pcm_bytes.empty());
	return audio;
}

DecodedAudio Decom_audio::decom_OGG(const std::filesystem::path& filepath) {
	DecodedAudio audio;
	std::ifstream in(filepath, std::ios::binary);
	if (!in) {
		return audio;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 28 || std::memcmp(buffer.data(), "OggS", 4) != 0) {
		return audio;
	}

	// Parse Vorbis Identification Header if in first pages
	// Ogg header: 0-3 "OggS", 4 version, 5 flags, 6-13 granule, 14-17 serial, 18-21 seq, 22-25 crc, 26 page_segments
	size_t pos = 0;
	const size_t len = buffer.size();

	while (pos + 27 < len) {
		if (std::memcmp(buffer.data() + pos, "OggS", 4) != 0) {
			pos++;
			continue;
		}

		uint8_t n_segments = buffer[pos + 26];
		if (pos + 27 + n_segments > len) {
			break;
		}

		size_t page_data_size = 0;
		for (size_t s = 0; s < n_segments; ++s) {
			page_data_size += buffer[pos + 27 + s];
		}

		size_t data_offset = pos + 27 + n_segments;
		if (data_offset + page_data_size > len) {
			break;
		}

		// Check for Vorbis Identification Header: "\x01vorbis"
		if (page_data_size >= 16 && std::memcmp(buffer.data() + data_offset, "\x01vorbis", 7) == 0) {
			audio.channels = buffer[data_offset + 11];
			audio.sample_rate = *reinterpret_cast<const uint32_t*>(buffer.data() + data_offset + 12);
			audio.bits_per_sample = 16;
		}

		// Collect audio payload packets
		audio.pcm_bytes.insert(audio.pcm_bytes.end(),
				       buffer.begin() + data_offset,
				       buffer.begin() + data_offset + page_data_size);

		pos = data_offset + page_data_size;
	}

	if (audio.sample_rate == 0) audio.sample_rate = 44100;
	if (audio.channels == 0) audio.channels = 2;
	audio.is_valid = !audio.pcm_bytes.empty();

	return audio;
}

DecodedAudio Decom_audio::decom_MP3(const std::filesystem::path& filepath) {
	DecodedAudio audio;
	std::ifstream in(filepath, std::ios::binary);
	if (!in) {
		return audio;
	}

	std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)),
				    std::istreambuf_iterator<char>());
	if (buffer.size() < 10) {
		return audio;
	}

	size_t start_pos = 0;

	// Skip ID3v2 header if present
	if (std::memcmp(buffer.data(), "ID3", 3) == 0) {
		uint32_t tag_size = ((buffer[6] & 0x7F) << 21) |
				    ((buffer[7] & 0x7F) << 14) |
				    ((buffer[8] & 0x7F) << 7) |
				    (buffer[9] & 0x7F);
		start_pos = 10 + tag_size;
	}

	if (start_pos < buffer.size()) {
		audio.pcm_bytes.insert(audio.pcm_bytes.end(),
				       buffer.begin() + start_pos,
				       buffer.end());
		audio.sample_rate = 44100;
		audio.channels = 2;
		audio.bits_per_sample = 16;
		audio.is_valid = true;
	}

	return audio;
}

DecodedAudio Decom_audio::decode_audio(const std::filesystem::path& filepath) {
	IdentifiedFile info = Identifier::identify(filepath);
	switch (info.format) {
	case FileFormat::WAV:
		return decom_WAV(filepath);
	case FileFormat::MP3:
		return decom_MP3(filepath);
	case FileFormat::OGG:
	default:
		return decom_OGG(filepath);
	}
}
