#include "decompressor/decompress_document.hpp"
#include "Identifier.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::string DecompressDocument::decompress_text(const std::filesystem::path& file_path) {
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return "";
	}

	std::ostringstream ss;
	ss << in.rdbuf();
	std::string content = ss.str();

	// Strip UTF-8 BOM if present
	if (content.size() >= 3 && static_cast<unsigned char>(content[0]) == 0xEF &&
	    static_cast<unsigned char>(content[1]) == 0xBB &&
	    static_cast<unsigned char>(content[2]) == 0xBF) {
		content.erase(0, 3);
	}

	return content;
}

std::string DecompressDocument::decompress_markdown(const std::filesystem::path& file_path) {
	return decompress_text(file_path);
}

std::string DecompressDocument::decompress_pdf(const std::filesystem::path& file_path) {
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return "";
	}

	std::vector<char> buffer((std::istreambuf_iterator<char>(in)),
				 std::istreambuf_iterator<char>());
	if (buffer.empty()) {
		return "";
	}

	std::ostringstream extracted;
	const size_t n = buffer.size();
	bool found_bt_et = false;

	// 1. Search for Begin Text (BT) and End Text (ET) blocks
	for (size_t i = 0; i + 2 <= n; ++i) {
		if (buffer[i] == 'B' && buffer[i + 1] == 'T' &&
		    (i == 0 || std::isspace(static_cast<unsigned char>(buffer[i - 1]))) &&
		    (i + 2 == n || std::isspace(static_cast<unsigned char>(buffer[i + 2])))) {

			found_bt_et = true;
			size_t j = i + 2;

			// Scan until matching ET
			while (j + 2 <= n) {
				if (buffer[j] == 'E' && buffer[j + 1] == 'T' &&
				    std::isspace(static_cast<unsigned char>(buffer[j - 1])) &&
				    (j + 2 == n || std::isspace(static_cast<unsigned char>(buffer[j + 2])))) {
					break;
				}

				// String literal: (text)
				if (buffer[j] == '(') {
					j++;
					std::string literal;
					int depth = 1;
					while (j < n && depth > 0) {
						if (buffer[j] == '\\' && j + 1 < n) {
							j++;
							switch (buffer[j]) {
							case 'n': literal += '\n'; break;
							case 'r': literal += '\r'; break;
							case 't': literal += '\t'; break;
							default: literal += buffer[j]; break;
							}
						} else if (buffer[j] == '(') {
							depth++;
							literal += '(';
						} else if (buffer[j] == ')') {
							depth--;
							if (depth > 0) literal += ')';
						} else {
							literal += buffer[j];
						}
						j++;
					}
					if (!literal.empty()) {
						extracted << literal << " ";
					}
				}
				// Hex string: <48656C6C6F>
				else if (buffer[j] == '<' && j + 1 < n && buffer[j + 1] != '<') {
					j++;
					std::string hex;
					while (j < n && buffer[j] != '>') {
						if (!std::isspace(static_cast<unsigned char>(buffer[j]))) {
							hex += buffer[j];
						}
						j++;
					}
					if (j < n && buffer[j] == '>') j++;

					std::string converted;
					for (size_t h = 0; h + 1 < hex.size(); h += 2) {
						try {
							std::string byte_str = hex.substr(h, 2);
							char ch = static_cast<char>(std::stoi(byte_str, nullptr, 16));
							if (std::isprint(static_cast<unsigned char>(ch)) || std::isspace(static_cast<unsigned char>(ch))) {
								converted += ch;
							}
						} catch (...) {
							break;
						}
					}
					if (!converted.empty()) {
						extracted << converted << " ";
					}
				} else {
					j++;
				}
			}
			i = j;
		}
	}

	// 2. Fallback: if no BT/ET blocks were found, extract printable sequences
	if (!found_bt_et || extracted.str().empty()) {
		std::string current_word;
		for (size_t i = 0; i < n; ++i) {
			unsigned char c = static_cast<unsigned char>(buffer[i]);
			if (std::isprint(c) && c != '<' && c != '>' && c != '[' && c != ']' &&
			    c != '/' && c != '{' && c != '}') {
				current_word += static_cast<char>(c);
			} else {
				if (current_word.size() >= 3) {
					// Ignore standard PDF syntax keywords
					if (current_word != "obj" && current_word != "endobj" &&
					    current_word != "stream" && current_word != "endstream" &&
					    current_word != "xref" && current_word != "trailer" &&
					    current_word != "startxref" && current_word != "FlateDecode") {
						extracted << current_word << " ";
					}
				}
				current_word.clear();
			}
		}
	}

	return extracted.str();
}

std::string DecompressDocument::decompress_docx(const std::filesystem::path& file_path) {
	std::ifstream in(file_path, std::ios::binary);
	if (!in) {
		return "";
	}

	std::vector<char> buffer((std::istreambuf_iterator<char>(in)),
				 std::istreambuf_iterator<char>());
	if (buffer.empty()) {
		return "";
	}

	std::ostringstream extracted;
	const size_t n = buffer.size();

	// Scan buffer for XML <w:t> tags
	for (size_t i = 0; i + 4 < n; ++i) {
		// Detect paragraph break <w:p>
		if (buffer[i] == '<' && buffer[i + 1] == 'w' && buffer[i + 2] == ':' && buffer[i + 3] == 'p') {
			extracted << "\n";
		}

		// Detect text tag <w:t> or <w:t ...>
		if (buffer[i] == '<' && buffer[i + 1] == 'w' && buffer[i + 2] == ':' && buffer[i + 3] == 't') {
			size_t tag_close = i + 4;
			while (tag_close < n && buffer[tag_close] != '>') {
				tag_close++;
			}
			if (tag_close < n && buffer[tag_close] == '>') {
				size_t content_start = tag_close + 1;
				size_t end_tag = content_start;
				while (end_tag + 5 < n) {
					if (buffer[end_tag] == '<' && buffer[end_tag + 1] == '/' &&
					    buffer[end_tag + 2] == 'w' && buffer[end_tag + 3] == ':' &&
					    buffer[end_tag + 4] == 't' && buffer[end_tag + 5] == '>') {
						break;
					}
					end_tag++;
				}
				if (end_tag > content_start) {
					std::string inner(buffer.begin() + content_start, buffer.begin() + end_tag);
					extracted << inner << " ";
				}
				i = end_tag + 5;
			}
		}
	}

	return extracted.str();
}

std::string DecompressDocument::extract_text(const std::filesystem::path& file_path) {
	IdentifiedFile info = Identifier::identify(file_path);
	switch (info.format) {
	case FileFormat::PDF:
		return decompress_pdf(file_path);
	case FileFormat::DOCX:
		return decompress_docx(file_path);
	case FileFormat::MD:
		return decompress_markdown(file_path);
	case FileFormat::TXT:
	case FileFormat::JSON:
	case FileFormat::CSV:
	case FileFormat::LOG:
	case FileFormat::CODE:
	default:
		return decompress_text(file_path);
	}
}
