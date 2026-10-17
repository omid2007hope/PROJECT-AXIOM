#ifndef include_identifier
#define include_identifier

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

enum class FileFormat {
	// Text & Document
	TXT,
	MD,
	JSON,
	CSV,
	LOG,
	CODE, // .cpp, .hpp, .c, .h, .cu, .cuh, .py, .js, .ts, etc.
	PDF,
	DOCX,

	// Images
	JPEG,
	PNG,
	BMP,
	WEBP,
	GIF,
	TIFF,

	// Audio
	WAV,
	OGG,
	MP3,
	FLAC,
	AAC,

	// Video
	MP4,
	MKV,
	AVI,
	MOV,
	WEBM,

	// 3D Models
	OBJ,
	STL,
	GLTF,
	GLB,

	// Archives & Containers
	ZIP,
	TAR,
	GZ,

	// Fallback
	UNKNOWN_BINARY,
	UNKNOWN
};

enum class ModalityCategory {
	DOCUMENT, // Pure text, code, PDF, DOCX -> Handled by Decom_doc
	IMAGE,    // Discrete visual pixels -> Handled by Decom_image
	VIDEO,    // Sequential video frames -> Handled by Decom_video
	AUDIO,    // Linear PCM waveforms -> Handled by Decom_audio
	THREE_D,  // 3D coordinates & meshes
	ARCHIVE,  // Container archives
	UNKNOWN
};

struct IdentifiedFile {
	std::filesystem::path path {};
	std::string filename {};
	std::string extension {};
	FileFormat format = FileFormat::UNKNOWN;
	ModalityCategory modality = ModalityCategory::UNKNOWN;
	uintmax_t size_bytes = 0;
	bool is_valid = false;
};

class Identifier {
public:
	// Identify a single file path given by the user
	static IdentifiedFile identify(const std::filesystem::path& filepath);

	// Identify all files under a directory (e.g., asset_sample/)
	static std::vector<IdentifiedFile> identify_directory(
		const std::filesystem::path& dir_path,
		bool recursive = true);

	// Classification and string conversions
	static ModalityCategory get_modality(FileFormat format);
	static std::string format_to_string(FileFormat format);
	static std::string modality_to_string(ModalityCategory modality);

	static bool is_document(FileFormat format);
	static bool is_image(FileFormat format);
	static bool is_audio(FileFormat format);
	static bool is_video(FileFormat format);

private:
	static FileFormat detect_magic_bytes(const uint8_t* header, size_t len, const std::string& ext);
	static FileFormat detect_by_extension(const std::string& ext);
	static std::string to_lower(std::string str);
};

#endif // include_identifier