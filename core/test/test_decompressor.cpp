#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/decompressor/decompress_audio.hpp"
#include "../include/decompressor/decompress_document.hpp"
#include "../include/decompressor/decompress_image.hpp"
#include "../include/decompressor/decompress_video.hpp"
#include "test_header.hpp"

bool test_decompressor() {
	int pass_count = 0;
	int fail_count = 0;

	std::cout << "\n=== Multimodal Decompressor Test Suite ===\n\n";

	std::filesystem::path temp_dir = test_core_dir() / "test_temp_decom";
	std::filesystem::create_directories(temp_dir);
	std::filesystem::path asset_dir = test_asset_dirs()[0];

	// ─── [1] Document Decompressor (DecompressDocument) ──────────────────────────
	std::cout << "[1] Document Decompressor (DecompressDocument: TXT, MD, PDF, DOCX)\n";

	// 1.1 TXT / MD extraction
	{
		std::filesystem::path txt_file = temp_dir / "sample.txt";
		{
			std::ofstream out(txt_file);
			out << "Silicon native computation with Zero Overhead.";
		}
		std::string txt_content = DecompressDocument::decompress_text(txt_file);
		test_check(txt_content.find("Silicon") != std::string::npos, "decompress_text extracts raw text content", pass_count, fail_count);
		test_check(txt_content.find("Overhead") != std::string::npos, "decompress_text preserves all vocabulary words", pass_count, fail_count);
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
		std::string pdf_content = DecompressDocument::decompress_pdf(pdf_file);
		test_check(pdf_content.find("Quantum Mechanics") != std::string::npos, "decompress_pdf extracts text from (text) Tj operators", pass_count, fail_count);
		test_check(pdf_content.find("Wavefunction") != std::string::npos, "decompress_pdf extracts multiple stream tokens", pass_count, fail_count);
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
		std::string docx_content = DecompressDocument::decompress_docx(docx_file);
		test_check(docx_content.find("Gravitational Constant") != std::string::npos, "decompress_docx extracts text enclosed in <w:t> tags", pass_count, fail_count);
		test_check(docx_content.find("9.80665") != std::string::npos, "decompress_docx extracts numerical and equation tokens", pass_count, fail_count);
	}

	// ─── [2] Image Decompressor (DecompressImage) ───────────────────────────────
	std::cout << "\n[2] Image Decompressor (DecompressImage: BMP, JPG, PNG)\n";

	// 2.1 Synthetic BMP 24-bit generation and bit-for-bit decode
	{
		std::filesystem::path bmp_file = temp_dir / "test_pattern.bmp";
		{
			std::ofstream out(bmp_file, std::ios::binary);
			uint32_t width = 4, height = 4;
			uint32_t row_stride = ((width * 24 + 31) / 32) * 4; // 12 bytes (4 pixels * 3 bytes, 0 pad)
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

		DecodedImage decoded = DecompressImage::decompress_bmp(bmp_file);
		test_check(decoded.is_valid, "decompress_bmp parses valid BMP", pass_count, fail_count);
		test_check_eq(decoded.width, 4, "decompress_bmp width is 4", pass_count, fail_count);
		test_check_eq(decoded.height, 4, "decompress_bmp height is 4", pass_count, fail_count);
		test_check_eq(decoded.channels, 3, "decompress_bmp channels is 3 (RGB)", pass_count, fail_count);
		test_check_eq(static_cast<int>(decoded.pixels[0]), 0xFF, "decompress_bmp pixel 0 is Red (0xFF)", pass_count, fail_count);
		test_check_eq(static_cast<int>(decoded.pixels[1]), 0x00, "decompress_bmp pixel 1 is Green (0x00)", pass_count, fail_count);
		test_check_eq(static_cast<int>(decoded.pixels[2]), 0x00, "decompress_bmp pixel 2 is Blue (0x00)", pass_count, fail_count);
	}

	// 2.2 Live Photo.jpeg asset decode
	{
		std::filesystem::path photo_path = asset_dir / "Photo.jpeg";
		if (std::filesystem::exists(photo_path)) {
			DecodedImage decoded = DecompressImage::decode_image(photo_path);
			test_check(decoded.is_valid, "decode_image successfully decodes Photo.jpeg", pass_count, fail_count);
			test_check(decoded.width > 0 && decoded.height > 0, "Photo.jpeg resolution extracted from SOF marker", pass_count, fail_count);
			test_check(!decoded.pixels.empty(), "Photo.jpeg pixel byte array populated", pass_count, fail_count);
		}
	}

	// ─── [3] Audio Decompressor (DecompressAudio) ───────────────────────────────
	std::cout << "\n[3] Audio Decompressor (DecompressAudio: WAV, OGG, MP3)\n";

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

		DecodedAudio decoded = DecompressAudio::decompress_wav(wav_file);
		test_check(decoded.is_valid, "decompress_wav parses valid RIFF/WAVE container", pass_count, fail_count);
		test_check_eq(decoded.sample_rate, 44100, "decompress_wav sample rate is 44100 Hz", pass_count, fail_count);
		test_check_eq(decoded.channels, 2, "decompress_wav channels is 2 (Stereo)", pass_count, fail_count);
		test_check_eq(decoded.bits_per_sample, 16, "decompress_wav bits per sample is 16-bit", pass_count, fail_count);
		test_check_eq(decoded.pcm_bytes.size(), static_cast<size_t>(128), "decompress_wav extracted exact 128 raw PCM bytes", pass_count, fail_count);
	}

	// 3.2 Live Voice.ogg asset decode
	{
		std::filesystem::path voice_path = asset_dir / "Voice.ogg";
		if (std::filesystem::exists(voice_path)) {
			DecodedAudio decoded = DecompressAudio::decode_audio(voice_path);
			test_check(decoded.is_valid, "decode_audio successfully decodes Voice.ogg", pass_count, fail_count);
			test_check(decoded.sample_rate > 0, "Voice.ogg sample rate extracted", pass_count, fail_count);
			test_check(!decoded.pcm_bytes.empty(), "Voice.ogg linear PCM samples populated", pass_count, fail_count);
		}
	}

	// ─── [4] Video Decompressor (DecompressVideo) ───────────────────────────────
	std::cout << "\n[4] Video Decompressor (DecompressVideo: MP4, MKV)\n";

	// 4.1 Live Video.mp4 asset decode
	{
		std::filesystem::path video_path = asset_dir / "Video.mp4";
		if (std::filesystem::exists(video_path)) {
			DecodedVideo decoded = DecompressVideo::decode_video(video_path);
			test_check(decoded.is_valid, "decode_video successfully parses Video.mp4", pass_count, fail_count);
			test_check(decoded.width > 0 && decoded.height > 0, "Video.mp4 resolution extracted from MP4 atoms", pass_count, fail_count);
			test_check(!decoded.raw_frames.empty(), "Video.mp4 frame bytes extracted from mdat box", pass_count, fail_count);
		}
	}

	// Clean up temp directory
	std::error_code ec;
	std::filesystem::remove_all(temp_dir, ec);

	std::cout << "=== Multimodal Decompressor Test Suite: " << pass_count << " passed, " << fail_count << " failed ===\n";
	return (fail_count == 0);
}
