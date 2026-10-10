#ifndef include_logistic
#define include_logistic

#include <cstddef>
#include <string>
#include <vector>

struct Model {

	std::string model_name = "";
	// X * Dim = 512
	size_t in_dim = 0;
	// Y * Dim = 4
	size_t out_dim = 0;
	// W
	std::vector<float> model_weights = {};
	// B
	std::vector<float> model_biases = {};
};


class Logistic {
private:
public:
	// Saves a model
	bool save_model(const Model& model, const std::string& path = "database/models");

	// Loads a model
	bool load_model(Model& model, const std::string& path = "database/models");
};

#endif