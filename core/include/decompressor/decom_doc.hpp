#ifndef include_decom_doc
#define include_decom_doc

#include <filesystem>
#include <string>

class Decom_doc {
public:
	// Master text extractor: inspects format and extracts clean UTF-8 text & math
	static std::string extract_text(const std::filesystem::path& filepath);

	// Individual document format decompressors
	static std::string decom_PDF(const std::filesystem::path& filepath);
	static std::string decom_DOCX(const std::filesystem::path& filepath);
	static std::string decom_TXT(const std::filesystem::path& filepath);
	static std::string decom_MD(const std::filesystem::path& filepath);
};

#endif // include_decom_doc