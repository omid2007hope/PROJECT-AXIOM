#include <string>
#include <vector>

// Just a prototype now!

// Raw data samples

// Text by user
std::vector<std::string> data_txt = {"Hello", "world", "deep", "learning"};
// Number by user
std::vector<double> data_num = {42.0, -3.14, 100.5, 0.001};
// Image by user
std::vector<double> data_img = {255.0, 128.0, 64.0, 0.0};
// Video by user
std::vector<double> data_video = {0.1, 0.5, 0.9, -0.2, 0.4};
// Voice by user
std::vector<double> data_voice = {0.02, -0.45, 0.88, -0.12, 0.05};

// Parsed data
std::vector<double> input = {};

/*
Turn raw data samples into the parsed input vector;
*/
void data_parser() {}