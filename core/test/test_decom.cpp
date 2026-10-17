#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/decompressor/decom_audio.hpp"
#include "../include/decompressor/decom_doc.hpp"
#include "../include/decompressor/decom_image.hpp"
#include "../include/decompressor/decom_video.hpp"
#include "test_header.hpp"

bool test_decom() {
	int s_pass = 0;
	int s_fail = 0;

	std::cout << "\n=== Multimodal Decompressor Test Suite ===\n\n";

	std::filesystem::path temp_dir = test_core_dir() / "test_temp_decom";
	std::filesystem::create_directories(temp_dir);
	std::filesystem::path asset_dir = test_asset_dirs()[0];

	// ─── [1] Document Decompressor (Decom_doc) ──────────────────────────────────
	std::cout << "[1] Document Decompressor (Decom_doc: TXT, MD, PDF, DOCX)\n";

	// 1.1 TXT / MD extraction
	{
		std::filesystem::path txt_file = temp_dir / "sample.txt";
		{
			std::ofstream out(txt_file);
			out << "Silicon native computation with Zero Overhead.";
		}
		std::string txt_content = Decom_doc::decom_TXT(txt_file);
		test_check(txt_content.find("Silicon") != std::string::npos, "decom_TXT extracts raw text content", s_pass, s_fail);
		test_check(txt_content.find("Overhead") != std::string::npos, "decom_TXT preserves all vocabulary words", s_pass, s_fail);
	}

	// 1.2 PDF text extraction (BT ... ET text operators)
	{
		std::filesystem::path pdf_file = temp_dir / "sample.pdf";
		{
			std::ofstream out(pdf_file, std::ios::binary);
			out << "%PDF-1.4\n"
			    << "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
			    << "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
			    << "3 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
			    << "4 0 obj << /Length 50 >> stream\n"
			    << "BT\n"
			    << "/F1 12 Tf\n"
			    << "(Quantum Mechanics) Tj\n"
			    << "(Wavefunction) Tj\n"
			    << "ET\n"
			    << "endstream endobj\n"
			    << "xref\ntrailer << /Root 1 0 R >>\nstartxref\n%%EOF";
		}
		std::string pdf_content = Decom_doc::decom_PDF(pdf_file);
		test_check(pdf_content.find("Quantum Mechanics") != std::string::npos, "decom_PDF extracts text from (text) Tj operators", s_pass, s_fail);
		test_check(pdf_content.find("Wavefunction") != std::string::npos, "decom_PDF extracts multiple stream tokens", s_pass, s_fail);
	}

	// 1.3 DOCX text extraction (<w:t> XML tags)
	{
		std::filesystem::path docx_file = temp_dir / "sample.docx";
		{
			std::ofstream out(docx_file, std::ios::binary);
			// Simulated DOCX XML chunk
			out << "PK\x03\x04"
			    << "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
			    << "<w:body>"
			    << "<w:p><w:r><w:t>Gravitational Constant</w:t></w:r></w:p>"
			    << "<w:p><w:r><w:t>9.80665</w:t></w:r></w:p>"
			    << "</w:body></w:document>";
		}
		std::string docx_content = Decom_doc::decom_DOCX(docx_file);
		test_check(docx_content.find("Gravitational Constant") != std::string::npos, "decom_DOCX extracts text enclosed in <w:t> tags", s_pass, s_fail);
		test_check(docx_content.find("9.80665") != std::string::npos, "decom_DOCX extracts numerical and equation tokens", s_pass, s_fail);
	}

	// ─── [2] Image Decompressor (Decom_image) ───────────────────────────────────
	std::cout << "\n[2] Image Decompressor (Decom_image: BMP, JPG, PNG)\n";

	// 2.1 Synthetic BMP 24-bit generation and bit-for-bit decode
	{
		std::filesystem::path bmp_file = temp_dir / "test_pattern.bmp";
		{
			std::ofstream out(bmp_file, std::ios::binary);
			uint32_t width = 4, height = 4;
			uint32_t row_stride = 16; // 4 * 3 = 12 + 4 pad = 16
			uint32_t pixel_offset = 54;
			uint32_t file_size = pixel_offset + row_stride * height;

			// File header (14 bytes)
			uint8_t file_hdr[14] = {'B', 'M'};
			*reinterpret_cast<uint32_t*>(&file_hdr[2]) = file_size;
			*reinterpret_cast<uint32_t*>(&file_hdr[10]) = pixel_offset;
			out.write(reinterpret_cast<char*>(file_hdr), 14);

			// Info header (40 bytes)
			uint8_t info_hdr[40] = {0};
			*reinterpret_cast<uint32_t*>(&info_hdr[0]) = 40;
			*reinterpret_cast<int32_t*>(&info_hdr[4]) = width;
			*reinterpret_cast<int32_t*>(&info_hdr[8]) = height;
			*reinterpret_cast<uint16_t*>(&info_hdr[12]) = 1;
			*reinterpret_cast<uint16_t*>(&info_hdr[14]) = 24; // 24-bit RGB
			out.write(reinterpret_cast<char*>(info_hdr), 40);

			// Pixel data: write 4x4 pure red pixels (BGR: 0x00, 0x00, 0xFF)
			std::vector<uint8_t> row(row_stride, 0);
			for (int x = 0; x < 4; ++x) {
				row[x * 3 + 0] = 0x00; // Blue
				row[x * 3 + 1] = 0x00; // Green
				row[x * 3 + 2] = 0xFF; // Red
			}
			for (int y = 0; y < 4; ++y) {
				out.write(reinterpret_cast<char*>(row.data()), row_stride);
			}
		}

		DecodedImage decoded = Decom_image::decom_BMP(bmp_file);
		test_check(decoded.is_valid, "decom_BMP parses valid BMP", s_pass, s_fail);
		test_check_eq(decoded.width, 4, "decom_BMP width is 4", s_pass, s_fail);
		test_check_eq(decoded.height, 4, "decom_BMP height is 4", s_pass, s_fail);
		test_check_eq(decoded.channels, 3, "decom_BMP channels is 3 (RGB)", s_pass, s_fail);
		test_check_eq(static_cast<int>(decoded.pixels[0]), 0xFF, "decom_BMP pixel 0 is Red (0xFF)", s_pass, s_fail);
		test_check_eq(static_cast<int>(decoded.pixels[1]), 0x00, "decom_BMP pixel 1 is Green (0x00)", s_pass, s_fail);
		test_check_eq(static_cast<int>(decoded.pixels[2]), 0x00, "decom_BMP pixel 2 is Blue (0x00)", s_pass, s_fail);
	}

	// 2.2 Live Photo.jpeg asset decode
	{
		std::filesystem::path photo_path = asset_dir / "Photo.jpeg";
		if (std::filesystem::exists(photo_path)) {
			DecodedImage decoded = Decom_image::decode_image(photo_path);
			test_check(decoded.is_valid, "decode_image successfully decodes Photo.jpeg", s_pass, s_fail);
			test_check(decoded.width > 0 && decoded.height > 0, "Photo.jpeg resolution extracted from SOF marker", s_pass, s_fail);
			test_check(!decoded.pixels.empty(), "Photo.jpeg pixel byte array populated", s_pass, s_fail);
		}
	}

	// ─── [3] Audio Decompressor (Decom_audio) ───────────────────────────────────
	std::cout << "\n[3] Audio Decompressor (Decom_audio: WAV, OGG, MP3)\n";

	// 3.1 Synthetic WAV generation and PCM extraction
	{
		std::filesystem::path wav_file = temp_dir / "test_tone.wav";
		{
			std::ofstream out(wav_file, std::ios::binary);
			uint32_t pcm_data_size = 128;
			uint32_t riff_size = 36 + pcm_data_size;

			out.write("RIFF", 4);
			out.write(reinterpret_cast<char*>(&riff_size), 4);
			out.write("WAVE", 4);

			out.write("fmt ", 4);
			uint32_t fmt_size = 16;
			uint16_t audio_fmt = 1; // PCM
			uint16_t channels = 2;
			uint32_t sample_rate = 44100;
			uint32_t byte_rate = 44100 * 2 * 2;
			uint16_t block_align = 4;
			uint16_t bits = 16;
			out.write(reinterpret_cast<char*>(&fmt_size), 4);
			out.write(reinterpret_cast<char*>(&audio_fmt), 2);
			out.write(reinterpret_cast<char*>(&channels), 2);
			out.write(reinterpret_cast<char*>(&sample_rate), 4);
			out.write(reinterpret_cast<char*>(&byte_rate), 4);
			out.write(reinterpret_cast<char*>(&block_align), 2);
			out.write(reinterpret_cast<char*>(&bits), 2);

			out.write("data", 4);
			out.write(reinterpret_cast<char*>(&pcm_data_size), 4);
			std::vector<uint8_t> dummy_samples(pcm_data_size, 0x7F);
			out.write(reinterpret_cast<char*>(dummy_samples.data()), pcm_data_size);
		}

		DecodedAudio decoded = Decom_audio::decom_WAV(wav_file);
		test_check(decoded.is_valid, "decom_WAV parses valid RIFF/WAVE container", s_pass, s_fail);
		test_check_eq(decoded.sample_rate, 44100, "decom_WAV sample rate is 44100 Hz", s_pass, s_fail);
		test_check_eq(decoded.channels, 2, "decom_WAV channels is 2 (Stereo)", s_pass, s_fail);
		test_check_eq(decoded.bits_per_sample, 16, "decom_WAV bits per sample is 16-bit", s_pass, s_fail);
		test_check_eq(decoded.pcm_bytes.size(), static_cast<size_t>(128), "decom_WAV extracted exact 128 raw PCM bytes", s_pass, s_fail);
	}

	// 3.2 Live Voice.ogg asset decode
	{
		std::filesystem::path voice_path = asset_dir / "Voice.ogg";
		if (std::filesystem::exists(voice_path)) {
			DecodedAudio decoded = Decom_audio::decode_audio(voice_path);
			test_check(decoded.is_valid, "decode_audio successfully decodes Voice.ogg", s_pass, s_fail);
			test_check(decoded.sample_rate > 0, "Voice.ogg sample rate extracted", s_pass, s_fail);
			test_check(!decoded.pcm_bytes.empty(), "Voice.ogg linear PCM samples populated", s_pass, s_fail);
		}
	}

	// ─── [4] Video Decompressor (Decom_video) ───────────────────────────────────
	std::cout << "\n[4] Video Decompressor (Decom_video: MP4, MKV)\n";

	// 4.1 Live Video.mp4 asset decode
	{
		std::filesystem::path video_path = asset_dir / "Video.mp4";
		if (std::filesystem::exists(video_path)) {
			DecodedVideo decoded = Decom_video::decode_video(video_path);
			test_check(decoded.is_valid, "decode_video successfully parses Video.mp4", s_pass, s_fail);
			test_check(decoded.width > 0 && decoded.height > 0, "Video.mp4 resolution extracted from MP4 atoms", s_pass, s_fail);
			test_check(!decoded.raw_frames.empty(), "Video.mp4 frame bytes extracted from mdat box", s_pass, s_fail);
		}
	}

	// Clean up temp directory
	std::error_code ec;
	std::filesystem::remove_all(temp_dir, ec);

	std::cout << "=== Multimodal Decompressor Test Suite: " << s_pass << " passed, " << s_fail << " failed ===\n";
	return (s_fail == 0);
}
