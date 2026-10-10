#include <cmath>
#include <iostream>
#include <stdfloat>
#include <vector>

#include "../include/ai.hpp"
#include "../include/optimizer.hpp"
#include "test_header.hpp"

bool test_ai() {
	std::cout << "\n=== AI Engine Test Suite ===\n";
	int s_pass = 0, s_fail = 0;

	const size_t in_dim = 512;
	const size_t out_dim = 4;
	const float w_min = -0.5f;
	const float w_max = 0.5f;

	Elements e {};

	// ── [1] Initialization & Dimensions ───────────────────────────────────────
	std::cout << "\n[1] AI Initialization & Dimensions\n";
	AI ai(in_dim, out_dim, w_min, w_max, e);

	test_check_eq(e.weight_elements, in_dim * out_dim, "Elements weight_elements == 2048", s_pass, s_fail);
	test_check_eq(e.bias_elements, out_dim, "Elements bias_elements == 4", s_pass, s_fail);
	test_check_eq(ai.h_weights.size(), in_dim * out_dim, "h_weights size == 2048", s_pass, s_fail);
	test_check_eq(ai.h_bias.size(), out_dim, "h_bias size == 4", s_pass, s_fail);

	// ── [2] Chaos / Symmetry Breaking ─────────────────────────────────────────
	std::cout << "\n[2] Weight Distribution & Chaos Verification\n";
	bool weights_in_bounds = true;
	float weight_sum = 0.0f;
	for (float w : ai.h_weights) {
		if (w < w_min || w > w_max) {
			weights_in_bounds = false;
			break;
		}
		weight_sum += w;
	}
	test_check(weights_in_bounds, "All weights bounded strictly in [w_min, w_max]", s_pass, s_fail);
	test_check(std::abs(weight_sum) > 0.001f, "Weights are non-zero chaotic distribution", s_pass, s_fail);

	bool biases_zero = true;
	for (float b : ai.h_bias) {
		if (b != 0.0f) {
			biases_zero = false;
			break;
		}
	}
	test_check(biases_zero, "Initial biases all set to 0.0f", s_pass, s_fail);

	// ── [3] GPU Transfer Roundtrip ────────────────────────────────────────────
	std::cout << "\n[3] AI GPU Transfer Roundtrip\n";
	test_check(ai.upload_weights_to_gpu(), "upload_weights_to_gpu() returns true", s_pass, s_fail);

	float original_w0 = ai.h_weights[0];
	ai.h_weights[0] = 999.0f; // Tamper with CPU cache

	test_check(ai.download_weights_from_gpu(), "download_weights_from_gpu() returns true", s_pass, s_fail);
	test_check(std::abs(ai.h_weights[0] - original_w0) < 1e-6f, "Downloaded weights restored original VRAM value", s_pass, s_fail);

	// ── [4] Loss Reading with Zero Steps ──────────────────────────────────────
	std::cout << "\n[4] Zero Pending Steps Handling\n";
	double zero_loss = 0.0;
	size_t zero_steps = 0;
	test_check(ai.read_loss(zero_loss, zero_steps), "read_loss() succeeds when pending_steps is 0", s_pass, s_fail);
	test_check_eq(zero_steps, size_t {0}, "steps reported is 0", s_pass, s_fail);
	test_check_eq(zero_loss, 0.0, "loss reported is 0.0", s_pass, s_fail);

	// ── [5] Training Loop Step & Loss Reading ─────────────────────────────────
	std::cout << "\n[5] Training Step Execution\n";
	const size_t RAM_BUF = 100;
	const size_t TOKEN_DIM = 4;
	const size_t CTX_WIN = 4; // 16 floats window, 4 floats target
	Portion portion(RAM_BUF, TOKEN_DIM, CTX_WIN);
	portion.allocate_pinned();

	// Populate ram buffer with synthetic tokens
	portion.ram_buffer_tokens.resize(RAM_BUF);
	for (size_t i = 0; i < RAM_BUF; ++i) {
		portion.ram_buffer_tokens[i] = static_cast<std::bfloat16_t>(0.1f * ((i % 10) + 1));
	}

	Optimizer optimizer(test_embedded_path(), 0.01f);

	TrainStatus status = ai.train(optimizer, portion);
	test_check(status == TrainStatus::Trained, "ai.train() returns TrainStatus::Trained on populated buffer", s_pass, s_fail);

	double train_loss = 0.0;
	size_t train_steps = 0;
	test_check(ai.read_loss(train_loss, train_steps), "read_loss() after train() succeeds", s_pass, s_fail);
	test_check_eq(train_steps, size_t {1}, "read_loss() reports 1 pending step", s_pass, s_fail);
	test_check(train_loss > 0.0, "Training loss is strictly positive", s_pass, s_fail);

	portion.free_pinned();

	// ── [6] Buffer Exhaustion (TrainStatus::NoData) ───────────────────────────
	std::cout << "\n[6] Buffer Exhaustion Handling\n";
	Portion empty_portion(RAM_BUF, TOKEN_DIM, CTX_WIN);
	empty_portion.allocate_pinned();
	// ram_buffer_tokens is empty
	TrainStatus empty_status = ai.train(optimizer, empty_portion);
	test_check(empty_status == TrainStatus::NoData, "ai.train() returns TrainStatus::NoData when buffer is empty", s_pass, s_fail);
	empty_portion.free_pinned();

	// ── [7] Multi-Step Training Convergence ───────────────────────────────────
	std::cout << "\n[7] Multi-Step Convergence\n";
	Portion conv_portion(RAM_BUF, TOKEN_DIM, CTX_WIN);
	conv_portion.allocate_pinned();

	auto refill = [&]() {
		conv_portion.ram_buffer_tokens.resize(RAM_BUF);
		for (size_t i = 0; i < RAM_BUF; ++i) {
			conv_portion.ram_buffer_tokens[i] = static_cast<std::bfloat16_t>(0.2f * ((i % 5) + 1));
		}
	};
	refill();

	// Measure step 1 loss
	ai.train(optimizer, conv_portion);
	double initial_loss = 0.0;
	size_t initial_steps = 0;
	ai.read_loss(initial_loss, initial_steps);

	// Train 40 additional steps
	for (int step = 0; step < 40; ++step) {
		if (ai.train(optimizer, conv_portion) == TrainStatus::NoData) {
			refill();
			ai.train(optimizer, conv_portion);
		}
	}

	double final_loss = 0.0;
	size_t final_steps = 0;
	ai.read_loss(final_loss, final_steps);

	std::cout << "  Initial step loss: " << initial_loss << " -> Multi-step loss: " << final_loss << "\n";
	test_check(final_loss < initial_loss, "Loss decreased after training (AI is actively learning)", s_pass, s_fail);

	conv_portion.free_pinned();

	std::cout << "=== AI Engine Test Suite: " << s_pass << " passed, " << s_fail
		  << " failed ===\n";
	return s_fail == 0;
}
