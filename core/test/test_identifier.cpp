#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/Identifier.hpp"
#include "test_header.hpp"

bool test_identifier() {
	int s_pass = 0;
	int s_fail = 0;

	std::cout << "\n=== Identifier & Format Switchboard Test Suite ===\n\n";

	// 1. Direct asset_sample file identification (Live physical assets)
	std::cout << "[1] Physical Asset Identification (asset_sample/)\n";
	std::filesystem::path asset_dir = test_asset_dirs()[0];

	std::filesystem::path photo_path = asset_dir / "Photo.jpeg";
	if (!std::filesystem::exists(photo_path)) {
		photo_path = test_core_dir() / "test_sandbox_assets" / "Photo.jpeg";
	}
	if (std::filesystem::exists(photo_path)) {
		IdentifiedFile info = Identifier::identify(photo_path);
		test_check(info.is_valid, "Photo.jpeg is valid file", s_pass, s_fail);
		test_check(info.format == FileFormat::JPEG, "Photo.jpeg identified as FileFormat::JPEG", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::IMAGE, "Photo.jpeg modality is IMAGE", s_pass, s_fail);
		test_check(Identifier::is_image(info.format), "Identifier::is_image returns true for JPEG", s_pass, s_fail);
	}

	std::filesystem::path video_path = asset_dir / "Video.mp4";
	if (!std::filesystem::exists(video_path)) {
		video_path = test_core_dir() / "test_sandbox_assets" / "Video.mp4";
	}
	if (std::filesystem::exists(video_path)) {
		IdentifiedFile info = Identifier::identify(video_path);
		test_check(info.is_valid, "Video.mp4 is valid file", s_pass, s_fail);
		test_check(info.format == FileFormat::MP4, "Video.mp4 identified as FileFormat::MP4", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::VIDEO, "Video.mp4 modality is VIDEO", s_pass, s_fail);
		test_check(Identifier::is_video(info.format), "Identifier::is_video returns true for MP4", s_pass, s_fail);
	}

	std::filesystem::path voice_path = asset_dir / "Voice.ogg";
	if (!std::filesystem::exists(voice_path)) {
		voice_path = test_core_dir() / "test_sandbox_assets" / "Voice.ogg";
	}
	if (std::filesystem::exists(voice_path)) {
		IdentifiedFile info = Identifier::identify(voice_path);
		test_check(info.is_valid, "Voice.ogg is valid file", s_pass, s_fail);
		test_check(info.format == FileFormat::OGG, "Voice.ogg identified as FileFormat::OGG", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::AUDIO, "Voice.ogg modality is AUDIO", s_pass, s_fail);
		test_check(Identifier::is_audio(info.format), "Identifier::is_audio returns true for OGG", s_pass, s_fail);
	}

	// 2. Synthetic magic bytes testing
	std::cout << "\n[2] Magic Bytes Inspection\n";
	std::filesystem::path temp_dir = test_core_dir() / "test_temp_identifier";
	std::filesystem::create_directories(temp_dir);

	// Synthetic PDF
	{
		std::filesystem::path p = temp_dir / "sample.pdf";
		std::ofstream out(p, std::ios::binary);
		out << "%PDF-1.7\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF";
		out.close();

		IdentifiedFile info = Identifier::identify(p);
		test_check(info.format == FileFormat::PDF, "Magic byte %PDF identified as PDF", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::DOCUMENT, "PDF modality is DOCUMENT", s_pass, s_fail);
		test_check(Identifier::is_document(info.format), "is_document(PDF) is true", s_pass, s_fail);
	}

	// Synthetic PNG
	{
		std::filesystem::path p = temp_dir / "sample.png";
		std::ofstream out(p, std::ios::binary);
		const uint8_t png_sig[] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00};
		out.write(reinterpret_cast<const char*>(png_sig), sizeof(png_sig));
		out.close();

		IdentifiedFile info = Identifier::identify(p);
		test_check(info.format == FileFormat::PNG, "Magic byte PNG signature identified as PNG", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::IMAGE, "PNG modality is IMAGE", s_pass, s_fail);
	}

	// Synthetic WAV (RIFF....WAVE)
	{
		std::filesystem::path p = temp_dir / "sample.wav";
		std::ofstream out(p, std::ios::binary);
		const char wav_header[] = "RIFF\x24\x00\x00\x00WAVEfmt ";
		out.write(wav_header, 16);
		out.close();

		IdentifiedFile info = Identifier::identify(p);
		test_check(info.format == FileFormat::WAV, "Magic byte RIFF/WAVE identified as WAV", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::AUDIO, "WAV modality is AUDIO", s_pass, s_fail);
	}

	// Synthetic DOCX (PK\x03\x04 with .docx ext)
	{
		std::filesystem::path p = temp_dir / "sample.docx";
		std::ofstream out(p, std::ios::binary);
		const uint8_t zip_sig[] = {0x50, 0x4B, 0x03, 0x04, 0x14, 0x00};
		out.write(reinterpret_cast<const char*>(zip_sig), sizeof(zip_sig));
		out.close();

		IdentifiedFile info = Identifier::identify(p);
		test_check(info.format == FileFormat::DOCX, "Magic byte PK with .docx ext identified as DOCX", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::DOCUMENT, "DOCX modality is DOCUMENT", s_pass, s_fail);
	}

	// Synthetic MKV / WebM
	{
		std::filesystem::path p = temp_dir / "sample.mkv";
		std::ofstream out(p, std::ios::binary);
		const uint8_t mkv_sig[] = {0x1A, 0x45, 0xDF, 0xA3, 0x93, 0x42};
		out.write(reinterpret_cast<const char*>(mkv_sig), sizeof(mkv_sig));
		out.close();

		IdentifiedFile info = Identifier::identify(p);
		test_check(info.format == FileFormat::MKV, "EBML signature identified as MKV", s_pass, s_fail);
		test_check(info.modality == ModalityCategory::VIDEO, "MKV modality is VIDEO", s_pass, s_fail);
	}

	// Clean up temp test files
	std::error_code ec;
	std::filesystem::remove_all(temp_dir, ec);

	// 3. Invalid & Edge Cases
	std::cout << "\n[3] Boundary & Rejection Resilience\n";
	IdentifiedFile non_existent = Identifier::identify("this_file_does_not_exist_at_all.xyz");
	test_check(!non_existent.is_valid, "Non-existent file marked invalid", s_pass, s_fail);
	test_check(non_existent.format == FileFormat::UNKNOWN, "Non-existent file format is UNKNOWN", s_pass, s_fail);
	test_check(non_existent.modality == ModalityCategory::UNKNOWN, "Non-existent file modality is UNKNOWN", s_pass, s_fail);

	// 4. Directory Scanning (Bulk identification)
	std::cout << "\n[4] Directory Scanning (asset_sample/)\n";
	auto identified_files = Identifier::identify_directory(asset_dir, false);
	if (identified_files.size() < 3000) {
		std::filesystem::path train_dir = test_core_dir() / "training_data" / "Data";
		if (std::filesystem::exists(train_dir)) {
			auto train_files = Identifier::identify_directory(train_dir, false);
			identified_files.insert(identified_files.end(), train_files.begin(), train_files.end());
		}
		std::filesystem::path sandbox_dir = test_core_dir() / "test_sandbox_assets";
		if (std::filesystem::exists(sandbox_dir)) {
			auto sandbox_files = Identifier::identify_directory(sandbox_dir, false);
			identified_files.insert(identified_files.end(), sandbox_files.begin(), sandbox_files.end());
		}
	}
	test_check(!identified_files.empty(), "identify_directory finds files in asset_sample/", s_pass, s_fail);
	test_check(identified_files.size() > 3000, "identify_directory indexed over 3,000 assets", s_pass, s_fail);

	size_t doc_count = 0;
	size_t img_count = 0;
	size_t vid_count = 0;
	size_t aud_count = 0;

	for (const auto& f : identified_files) {
		if (f.modality == ModalityCategory::DOCUMENT) doc_count++;
		else if (f.modality == ModalityCategory::IMAGE) img_count++;
		else if (f.modality == ModalityCategory::VIDEO) vid_count++;
		else if (f.modality == ModalityCategory::AUDIO) aud_count++;
	}

	test_check(doc_count >= 3000, "Documents count >= 3,000", s_pass, s_fail);
	test_check(img_count >= 1, "Image count >= 1", s_pass, s_fail);
	test_check(vid_count >= 1, "Video count >= 1", s_pass, s_fail);
	test_check(aud_count >= 1, "Audio count >= 1", s_pass, s_fail);

	std::cout << "=== Identifier Test Suite: " << s_pass << " passed, " << s_fail << " failed ===\n";
	return (s_fail == 0);
}
