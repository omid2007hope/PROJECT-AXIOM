#ifndef include_decompress_document
#define include_decompress_document

#include <filesystem>
#include <string>

class DecompressDocument {
public:
	// Master text extractor: inspects format and extracts clean UTF-8 text & math
	static std::string extract_text(const std::filesystem::path& file_path);

	// Individual document format decompressors
	static std::string decompress_pdf(const std::filesystem::path& file_path);
	static std::string decompress_docx(const std::filesystem::path& file_path);
	static std::string decompress_text(const std::filesystem::path& file_path);
	static std::string decompress_markdown(const std::filesystem::path& file_path);
};

#endif // include_decompress_document