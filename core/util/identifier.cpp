#include "Identifier.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>

std::string Identifier::to_lower(std::string str) {
	for (char& c : str) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	return str;
}

FileFormat Identifier::detect_magic_bytes(const uint8_t* header, size_t len, const std::string& ext) {
	if (len == 0) {
		return detect_by_extension(ext);
	}

	// 1. PDF: %PDF- (0x25, 0x50, 0x44, 0x46)
	if (len >= 4 && header[0] == 0x25 && header[1] == 0x50 && header[2] == 0x44 && header[3] == 0x46) {
		return FileFormat::PDF;
	}

	// 2. JPEG: 0xFF, 0xD8, 0xFF
	if (len >= 3 && header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF) {
		return FileFormat::JPEG;
	}

	// 3. PNG: 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A
	if (len >= 8 && header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47 &&
	    header[4] == 0x0D && header[5] == 0x0A && header[6] == 0x1A && header[7] == 0x0A) {
		return FileFormat::PNG;
	}

	// 4. GIF: GIF87a or GIF89a
	if (len >= 6 && header[0] == 'G' && header[1] == 'I' && header[2] == 'F' && header[3] == '8' &&
	    (header[4] == '7' || header[4] == '9') && header[5] == 'a') {
		return FileFormat::GIF;
	}

	// 5. BMP: 'B', 'M' (0x42, 0x4D)
	if (len >= 2 && header[0] == 0x42 && header[1] == 0x4D) {
		return FileFormat::BMP;
	}

	// 6. RIFF-based formats (WAV, WEBP, AVI)
	if (len >= 12 && header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F') {
		if (header[8] == 'W' && header[9] == 'E' && header[10] == 'B' && header[11] == 'P') {
			return FileFormat::WEBP;
		}
		if (header[8] == 'W' && header[9] == 'A' && header[10] == 'V' && header[11] == 'E') {
			return FileFormat::WAV;
		}
		if (header[8] == 'A' && header[9] == 'V' && header[10] == 'I' && header[11] == ' ') {
			return FileFormat::AVI;
		}
	}

	// 7. OGG: OggS (0x4F, 0x67, 0x67, 0x53)
	if (len >= 4 && header[0] == 0x4F && header[1] == 0x67 && header[2] == 0x67 && header[3] == 0x53) {
		return FileFormat::OGG;
	}

	// 8. FLAC: fLaC (0x66, 0x4C, 0x61, 0x43)
	if (len >= 4 && header[0] == 0x66 && header[1] == 0x4C && header[2] == 0x61 && header[3] == 0x43) {
		return FileFormat::FLAC;
	}

	// 9. MP3: ID3 header or MPEG sync bits
	if ((len >= 3 && header[0] == 'I' && header[1] == 'D' && header[2] == '3') ||
	    (len >= 2 && header[0] == 0xFF && (header[1] & 0xE0) == 0xE0)) {
		return FileFormat::MP3;
	}

	// 10. MP4: 'ftyp' at offset 4..7
	if (len >= 8 && header[4] == 'f' && header[5] == 't' && header[6] == 'y' && header[7] == 'p') {
		return FileFormat::MP4;
	}

	// 11. MKV / WebM: EBML header (0x1A, 0x45, 0xDF, 0xA3)
	if (len >= 4 && header[0] == 0x1A && header[1] == 0x45 && header[2] == 0xDF && header[3] == 0xA3) {
		if (ext == ".webm") {
			return FileFormat::WEBM;
		}
		return FileFormat::MKV;
	}

	// 12. ZIP / DOCX: PK\x03\x04 (0x50, 0x4B, 0x03, 0x04)
	if (len >= 4 && header[0] == 0x50 && header[1] == 0x4B && header[2] == 0x03 && header[3] == 0x04) {
		if (ext == ".docx") {
			return FileFormat::DOCX;
		}
		return FileFormat::ZIP;
	}

	// 13. GZIP: 0x1F, 0x8B
	if (len >= 2 && header[0] == 0x1F && header[1] == 0x8B) {
		return FileFormat::GZ;
	}

	// 14. TIFF: 'II' (0x49 0x49) or 'MM' (0x4D 0x4D)
	if (len >= 4 && ((header[0] == 0x49 && header[1] == 0x49 && header[2] == 0x2A && header[3] == 0x00) ||
			 (header[0] == 0x4D && header[1] == 0x4D && header[2] == 0x00 && header[3] == 0x2A))) {
		return FileFormat::TIFF;
	}

	// 15. Binary vs Text Detection Fallback
	bool has_null_byte = false;
	for (size_t i = 0; i < len; ++i) {
		if (header[i] == 0) {
			has_null_byte = true;
			break;
		}
	}

	if (has_null_byte) {
		FileFormat by_ext = detect_by_extension(ext);
		if (by_ext != FileFormat::UNKNOWN && by_ext != FileFormat::TXT) {
			return by_ext;
		}
		return FileFormat::UNKNOWN_BINARY;
	}

	// Clean text or code
	return detect_by_extension(ext);
}

FileFormat Identifier::detect_by_extension(const std::string& ext) {
	if (ext == ".txt" || ext == ".text") return FileFormat::TXT;
	if (ext == ".md" || ext == ".markdown") return FileFormat::MD;
	if (ext == ".json") return FileFormat::JSON;
	if (ext == ".csv") return FileFormat::CSV;
	if (ext == ".log") return FileFormat::LOG;
	if (ext == ".pdf") return FileFormat::PDF;
	if (ext == ".docx") return FileFormat::DOCX;

	// Source code & config formats
	if (ext == ".c" || ext == ".cpp" || ext == ".cc" || ext == ".cxx" ||
	    ext == ".h" || ext == ".hpp" || ext == ".cu" || ext == ".cuh" ||
	    ext == ".py" || ext == ".js" || ext == ".ts" || ext == ".rs" ||
	    ext == ".go" || ext == ".java" || ext == ".sh" || ext == ".bat" ||
	    ext == ".ps1" || ext == ".sql" || ext == ".html" || ext == ".htm" ||
	    ext == ".xml" || ext == ".yaml" || ext == ".yml" || ext == ".ini" ||
	    ext == ".cfg" || ext == ".conf") {
		return FileFormat::CODE;
	}

	// Images
	if (ext == ".jpg" || ext == ".jpeg") return FileFormat::JPEG;
	if (ext == ".png") return FileFormat::PNG;
	if (ext == ".bmp") return FileFormat::BMP;
	if (ext == ".webp") return FileFormat::WEBP;
	if (ext == ".gif") return FileFormat::GIF;
	if (ext == ".tif" || ext == ".tiff") return FileFormat::TIFF;

	// Audio
	if (ext == ".wav") return FileFormat::WAV;
	if (ext == ".ogg") return FileFormat::OGG;
	if (ext == ".mp3") return FileFormat::MP3;
	if (ext == ".flac") return FileFormat::FLAC;
	if (ext == ".aac" || ext == ".m4a") return FileFormat::AAC;

	// Video
	if (ext == ".mp4" || ext == ".m4v") return FileFormat::MP4;
	if (ext == ".mkv") return FileFormat::MKV;
	if (ext == ".avi") return FileFormat::AVI;
	if (ext == ".mov") return FileFormat::MOV;
	if (ext == ".webm") return FileFormat::WEBM;

	// 3D
	if (ext == ".obj") return FileFormat::OBJ;
	if (ext == ".stl") return FileFormat::STL;
	if (ext == ".gltf") return FileFormat::GLTF;
	if (ext == ".glb") return FileFormat::GLB;

	// Archive
	if (ext == ".zip") return FileFormat::ZIP;
	if (ext == ".tar") return FileFormat::TAR;
	if (ext == ".gz") return FileFormat::GZ;

	return FileFormat::UNKNOWN;
}

ModalityCategory Identifier::get_modality(FileFormat format) {
	switch (format) {
	case FileFormat::TXT:
	case FileFormat::MD:
	case FileFormat::JSON:
	case FileFormat::CSV:
	case FileFormat::LOG:
	case FileFormat::CODE:
	case FileFormat::PDF:
	case FileFormat::DOCX:
		return ModalityCategory::DOCUMENT;

	case FileFormat::JPEG:
	case FileFormat::PNG:
	case FileFormat::BMP:
	case FileFormat::WEBP:
	case FileFormat::GIF:
	case FileFormat::TIFF:
		return ModalityCategory::IMAGE;

	case FileFormat::MP4:
	case FileFormat::MKV:
	case FileFormat::AVI:
	case FileFormat::MOV:
	case FileFormat::WEBM:
		return ModalityCategory::VIDEO;

	case FileFormat::WAV:
	case FileFormat::OGG:
	case FileFormat::MP3:
	case FileFormat::FLAC:
	case FileFormat::AAC:
		return ModalityCategory::AUDIO;

	case FileFormat::OBJ:
	case FileFormat::STL:
	case FileFormat::GLTF:
	case FileFormat::GLB:
		return ModalityCategory::THREE_D;

	case FileFormat::ZIP:
	case FileFormat::TAR:
	case FileFormat::GZ:
		return ModalityCategory::ARCHIVE;

	case FileFormat::UNKNOWN_BINARY:
	case FileFormat::UNKNOWN:
	default:
		return ModalityCategory::UNKNOWN;
	}
}

std::string Identifier::format_to_string(FileFormat format) {
	switch (format) {
	case FileFormat::TXT: return "Text";
	case FileFormat::MD: return "Markdown";
	case FileFormat::JSON: return "JSON";
	case FileFormat::CSV: return "CSV";
	case FileFormat::LOG: return "Log";
	case FileFormat::CODE: return "Source Code";
	case FileFormat::PDF: return "PDF Document";
	case FileFormat::DOCX: return "DOCX Document";
	case FileFormat::JPEG: return "JPEG Image";
	case FileFormat::PNG: return "PNG Image";
	case FileFormat::BMP: return "BMP Image";
	case FileFormat::WEBP: return "WebP Image";
	case FileFormat::GIF: return "GIF Image";
	case FileFormat::TIFF: return "TIFF Image";
	case FileFormat::WAV: return "WAV Audio";
	case FileFormat::OGG: return "OGG Audio";
	case FileFormat::MP3: return "MP3 Audio";
	case FileFormat::FLAC: return "FLAC Audio";
	case FileFormat::AAC: return "AAC Audio";
	case FileFormat::MP4: return "MP4 Video";
	case FileFormat::MKV: return "MKV Video";
	case FileFormat::AVI: return "AVI Video";
	case FileFormat::MOV: return "MOV Video";
	case FileFormat::WEBM: return "WebM Video";
	case FileFormat::OBJ: return "Wavefront OBJ (3D)";
	case FileFormat::STL: return "STL Mesh (3D)";
	case FileFormat::GLTF: return "glTF (3D)";
	case FileFormat::GLB: return "Binary glTF (3D)";
	case FileFormat::ZIP: return "ZIP Archive";
	case FileFormat::TAR: return "TAR Archive";
	case FileFormat::GZ: return "GZip Archive";
	case FileFormat::UNKNOWN_BINARY: return "Raw Binary";
	case FileFormat::UNKNOWN:
	default:
		return "Unknown";
	}
}

std::string Identifier::modality_to_string(ModalityCategory modality) {
	switch (modality) {
	case ModalityCategory::DOCUMENT: return "Document/Text";
	case ModalityCategory::IMAGE: return "Image";
	case ModalityCategory::VIDEO: return "Video";
	case ModalityCategory::AUDIO: return "Audio/Voice";
	case ModalityCategory::THREE_D: return "3D Model";
	case ModalityCategory::ARCHIVE: return "Archive/Container";
	case ModalityCategory::UNKNOWN:
	default:
		return "Unknown";
	}
}

bool Identifier::is_document(FileFormat format) {
	return get_modality(format) == ModalityCategory::DOCUMENT;
}

bool Identifier::is_image(FileFormat format) {
	return get_modality(format) == ModalityCategory::IMAGE;
}

bool Identifier::is_audio(FileFormat format) {
	return get_modality(format) == ModalityCategory::AUDIO;
}

bool Identifier::is_video(FileFormat format) {
	return get_modality(format) == ModalityCategory::VIDEO;
}

IdentifiedFile Identifier::identify(const std::filesystem::path& filepath) {
	IdentifiedFile result;
	result.path = filepath;
	result.filename = filepath.filename().string();
	result.extension = to_lower(filepath.extension().string());

	std::error_code ec;
	if (!std::filesystem::exists(filepath, ec) || !std::filesystem::is_regular_file(filepath, ec)) {
		result.is_valid = false;
		result.format = FileFormat::UNKNOWN;
		result.modality = ModalityCategory::UNKNOWN;
		return result;
	}

	result.size_bytes = std::filesystem::file_size(filepath, ec);
	if (ec) {
		result.size_bytes = 0;
	}

	// Probe magic bytes
	std::ifstream file(filepath, std::ios::binary);
	if (!file) {
		result.is_valid = false;
		result.format = FileFormat::UNKNOWN;
		result.modality = ModalityCategory::UNKNOWN;
		return result;
	}

	uint8_t header[64] = {0};
	file.read(reinterpret_cast<char*>(header), sizeof(header));
	std::streamsize bytes_read = file.gcount();

	result.format = detect_magic_bytes(header, static_cast<size_t>(bytes_read), result.extension);
	result.modality = get_modality(result.format);
	result.is_valid = true;

	return result;
}

std::vector<IdentifiedFile> Identifier::identify_directory(
	const std::filesystem::path& dir_path,
	bool recursive) {
	std::vector<IdentifiedFile> results;

	std::error_code ec;
	if (!std::filesystem::exists(dir_path, ec) || !std::filesystem::is_directory(dir_path, ec)) {
		return results;
	}

	auto process_entry = [&](const std::filesystem::directory_entry& entry) {
		if (entry.is_regular_file(ec)) {
			// Skip hidden files, .git, and temporary files
			std::string fname = entry.path().filename().string();
			if (!fname.empty() && fname[0] == '.') {
				return;
			}
			results.push_back(identify(entry.path()));
		}
	};

	if (recursive) {
		for (std::filesystem::recursive_directory_iterator it(dir_path, ec), end; it != end; ++it) {
			if (ec) break;
			// Skip 'archive' or 'database' folders
			if (it->is_directory(ec)) {
				std::string dirname = it->path().filename().string();
				if (dirname == "archive" || dirname == "database" || dirname == ".git") {
					it.disable_recursion_pending();
					continue;
				}
			}
			process_entry(*it);
		}
	} else {
		for (std::filesystem::directory_iterator it(dir_path, ec), end; it != end; ++it) {
			if (ec) break;
			process_entry(*it);
		}
	}

	// Deterministic alphabetical sort
	std::sort(results.begin(), results.end(), [](const IdentifiedFile& a, const IdentifiedFile& b) {
		return a.path.string() < b.path.string();
	});

	return results;
}
