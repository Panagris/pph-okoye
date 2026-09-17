#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <omp.h>
#include <sys/stat.h>

#ifndef NO_OPENCV
#include <opencv2/opencv.hpp>
#endif

// ============================================================================
// Core function from convolve.cpp - Fast Laser Speckle Contrast calculation
// ============================================================================
extern "C" {
float fastSpeckle(float* input_data, int rows, int cols, int kernel_size)
{
	const int pad = (kernel_size - 1) / 2;
	
	// Precalculation of reciprocals for faster processing
	const float recip_2 = 1.0f / (kernel_size * (kernel_size - 1));
	const float recip_3 = 1.0f / kernel_size;

	// Accumulators
	double total = 0.0;
	double count = 0.0;

	const int stride = cols + 1;  // diagonal step (one down, one right)

	#pragma omp parallel for reduction(+:total,count) schedule(static)
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {

			// Determine start of the diagonal segment
			int start_y = y - pad;
			int start_x = x - pad;
			int end_y   = y + pad;
			int end_x   = x + pad;

			// Clamp to borders once
			if (start_y < 0) start_y = 0;
			if (start_x < 0) start_x = 0;
			if (end_y >= rows) end_y = rows - 1;
			if (end_x >= cols) end_x = cols - 1;

			// Compute effective length 
			int len = end_y - start_y + 1;
			
			float* ptr = input_data + start_y * cols + start_x;

			float acc = 0.0f;
			float acc_sq = 0.0f;

			// main diagonal accumulation
			for (int k = 0; k < len; ++k, ptr += stride) {
				float v = *ptr;
				acc += v;
				acc_sq += v * v;
			}

			float denom = acc * recip_3;
			if (denom != 0.0f) {
				float val = sqrtf((len * acc_sq - acc * acc) * recip_2) / denom;
				if (std::isfinite(val)) {
					total += val;
					count += 1.0;
				}
			}
		}
	}

	return (count == 0.0) ? NAN : (float)(total / count);
}
}

// ============================================================================
// Helper Utilities
// ============================================================================
bool fileExists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

bool parseResolution(const std::string& str, int& width, int& height) {
    size_t x_pos = str.find('x');
    if (x_pos != std::string::npos) {
        try {
            width = std::stoi(str.substr(0, x_pos));
            height = std::stoi(str.substr(x_pos + 1));
            return width > 0 && height > 0;
        } catch (...) {
            return false;
        }
    } else {
        try {
            int val = std::stoi(str);
            width = val;
            height = val;
            return width > 0;
        } catch (...) {
            return false;
        }
    }
}

std::string formatDouble(double val) {
    std::ostringstream ss;
    if (val == std::floor(val)) {
        ss << static_cast<int>(val);
    } else {
        ss << std::fixed << std::setprecision(1) << val;
    }
    return ss.str();
}

void printHelp(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <video_type> <resolution> [source_fps] [target_fps] [kernel_size] [exposure_time_ms]\n\n"
              << "Arguments:\n"
              << "  <video_type>       Speckle pattern contrast flavor: 'high', 'low', or 'random'\n"
              << "  <resolution>       Video resolution. E.g., '1000' (for 1000x1000) or '1024x1024' (for widthxheight)\n"
              << "  [source_fps]       Frame rate of the generated video (default: 80 FPS)\n"
              << "  [target_fps]       Target processing frame rate of the pipeline (default: 80 FPS)\n"
              << "  [kernel_size]      Spatial window size for fastSpeckle (default: 5)\n"
              << "  [exposure_time_ms] Target camera exposure time in ms (default: 3.0 ms)\n\n"
              << "Options:\n"
              << "  -h, --help         Show this detailed help menu and exit\n\n"
              << "Example:\n"
              << "  " << prog_name << " random 1024x1024 120 80 5 3.0\n"
              << "  This checks for an existing file named speckle_random_1024x1024_s120_t80_k5_e3.mp4.\n"
              << "  If it exists, it processes it immediately; otherwise, it generates it first.\n"
              << std::endl;
}

// Separable 3x3 box blur filter (representing physical speckle aperture blur)
void boxBlur3x3(const std::vector<float>& src, std::vector<float>& dst, int w, int h) {
    std::vector<float> temp(w * h);
    // Horizontal pass
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int dx = -1; dx <= 1; ++dx) {
                int xx = std::min(std::max(x + dx, 0), w - 1);
                sum += src[y * w + xx];
            }
            temp[y * w + x] = sum / 3.0f;
        }
    }
    // Vertical pass
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = std::min(std::max(y + dy, 0), h - 1);
                sum += temp[yy * w + x];
            }
            dst[y * w + x] = sum / 3.0f;
        }
    }
}

// ============================================================================
// Speckle Video Generation and Execution
// ============================================================================

#ifdef NO_OPENCV
// ----------------------------------------------------------------------------
// FFmpeg Pipe Fallback Implementation (No OpenCV dependencies)
// ----------------------------------------------------------------------------
void generateSpeckleVideo(const std::string& filename, const std::string& video_type, 
                          int width, int height, double source_fps, int total_frames) {
    // Open FFmpeg pipe for writing grayscale raw video, compressing to MP4 (yuv420p)
    std::string ffmpeg_cmd = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt gray -s " 
                            + std::to_string(width) + "x" + std::to_string(height) 
                            + " -r " + std::to_string(source_fps) 
                            + " -i - -an -vcodec libx264 -pix_fmt yuv420p " + filename + " >/dev/null 2>&1";
    FILE* pipe = popen(ffmpeg_cmd.c_str(), "w");
    if (!pipe) {
        std::cerr << "Error: Could not open FFmpeg write pipe." << std::endl;
        exit(1);
    }

    std::mt19937 prng(42); // Seeded for deterministic and repeatable patterns
    std::uniform_real_distribution<float> dist(0.0001f, 1.0f);

    // S is the static speckle component
    std::vector<float> S(width * height);
    for (int i = 0; i < width * height; ++i) {
        S[i] = -std::log(dist(prng));
    }
    std::vector<float> S_blurred(width * height);
    boxBlur3x3(S, S_blurred, width, height);

    std::vector<float> D(width * height);
    std::vector<float> D_blurred(width * height);
    std::vector<uint8_t> out_bytes(width * height);

    for (int f = 0; f < total_frames; ++f) {
        // Generate dynamic speckle component
        for (int i = 0; i < width * height; ++i) {
            D[i] = -std::log(dist(prng));
        }
        boxBlur3x3(D, D_blurred, width, height);

        float alpha = 0.5f;
        float scale = 0.5f;
        float offset = 0.0f;

        if (video_type == "high") {
            alpha = 0.1f;
            scale = 0.8f;
            offset = 0.1f;
        } else if (video_type == "low") {
            alpha = 0.9f;
            scale = 0.15f;
            offset = 0.85f;
        } else { // random-contrast (pulsatile blood flow emulation)
            float pulse = 0.5f + 0.3f * std::sin(2.0f * M_PI * f / source_fps);
            alpha = pulse;
            scale = 0.3f + 0.2f * std::cos(2.0f * M_PI * f / source_fps);
            offset = 0.5f - 0.1f * std::cos(2.0f * M_PI * f / source_fps);
        }

        for (int i = 0; i < width * height; ++i) {
            float val = (1.0f - alpha) * S_blurred[i] + alpha * D_blurred[i];
            val = val * scale + offset;
            int pixel = static_cast<int>(val * 255.0f);
            out_bytes[i] = static_cast<uint8_t>(std::min(std::max(pixel, 0), 255));
        }

        fwrite(out_bytes.data(), 1, out_bytes.size(), pipe);
    }

    pclose(pipe);
}

void processSpeckleVideo(const std::string& filename, int kernel_size, double target_fps, 
                         double exposure_time_ms, int width, int height, int total_frames) {
    // Open FFmpeg pipe for reading grayscale frames from MP4
    std::string ffmpeg_cmd = "ffmpeg -i " + filename + " -f image2pipe -vcodec rawvideo -pix_fmt gray - 2>/dev/null";
    FILE* pipe = popen(ffmpeg_cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: Could not open FFmpeg read pipe." << std::endl;
        exit(1);
    }

    std::vector<uint8_t> frame_buf(width * height);
    std::vector<float> float_frame(width * height);
    std::vector<double> frame_times_ms;
    frame_times_ms.reserve(total_frames);

    int processed_frames = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    while (fread(frame_buf.data(), 1, frame_buf.size(), pipe) == frame_buf.size()) {
        // Convert to single-precision float normalized to [0.0, 1.0]
        #pragma omp parallel for
        for (int i = 0; i < width * height; ++i) {
            float_frame[i] = static_cast<float>(frame_buf[i]) / 255.0f;
        }

        auto f_start = std::chrono::high_resolution_clock::now();
        float result = fastSpeckle(float_frame.data(), height, width, kernel_size);
        auto f_end = std::chrono::high_resolution_clock::now();

        double frame_time = std::chrono::duration<double, std::milli>(f_end - f_start).count();
        frame_times_ms.push_back(frame_time);

        processed_frames++;

        std::cout << "\rProcessing frame: " << processed_frames 
                  << " | Result: " << std::fixed << std::setprecision(5) << result 
                  << " | Time: " << std::setprecision(2) << frame_time << " ms" << std::flush;
    }

    pclose(pipe);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_wall_time_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n\n=== Performance Statistics ===" << std::endl;
    if (processed_frames > 0) {
        double min_time = frame_times_ms[0];
        double max_time = frame_times_ms[0];
        double sum_time = 0.0;

        for (double t : frame_times_ms) {
            if (t < min_time) min_time = t;
            if (t > max_time) max_time = t;
            sum_time += t;
        }

        double avg_time_ms = sum_time / processed_frames;
        double processing_fps = 1000.0 / avg_time_ms;
        size_t frame_bytes = width * height * sizeof(float);
        double throughput_gbps = (frame_bytes * processed_frames) / (total_wall_time_sec * 1e9);

        std::cout << "Frames Processed:  " << processed_frames << std::endl;
        std::cout << "Total Execution:   " << std::setprecision(3) << total_wall_time_sec << " s" << std::endl;
        std::cout << "Avg Kernel Time:   " << std::setprecision(3) << avg_time_ms << " ms/frame" << std::endl;
        std::cout << "Min Kernel Time:   " << std::setprecision(3) << min_time << " ms" << std::endl;
        std::cout << "Max Kernel Time:   " << std::setprecision(3) << max_time << " ms" << std::endl;
        std::cout << "Processing Speed:  " << std::setprecision(2) << processing_fps << " FPS" << std::endl;
        std::cout << "Data Throughput:   " << std::setprecision(3) << throughput_gbps << " GB/s" << std::endl;
    } else {
        std::cout << "No frames were processed." << std::endl;
    }
    std::cout << "==============================" << std::endl;
}

#else
// ----------------------------------------------------------------------------
// OpenCV Implementation (Standard target behavior)
// ----------------------------------------------------------------------------
void generateSpeckleVideo(const std::string& filename, const std::string& video_type, 
                          int width, int height, double source_fps, int total_frames) {
    cv::VideoWriter writer(filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), source_fps, cv::Size(width, height), true);
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open cv::VideoWriter with filename " << filename << std::endl;
        exit(1);
    }

    std::mt19937 prng(42);
    std::uniform_real_distribution<float> dist(0.0001f, 1.0f);

    std::vector<float> S(width * height);
    for (int i = 0; i < width * height; ++i) {
        S[i] = -std::log(dist(prng));
    }
    std::vector<float> S_blurred(width * height);
    boxBlur3x3(S, S_blurred, width, height);

    std::vector<float> D(width * height);
    std::vector<float> D_blurred(width * height);
    cv::Mat gray_frame(height, width, CV_8UC1);
    cv::Mat color_frame;

    for (int f = 0; f < total_frames; ++f) {
        for (int i = 0; i < width * height; ++i) {
            D[i] = -std::log(dist(prng));
        }
        boxBlur3x3(D, D_blurred, width, height);

        float alpha = 0.5f;
        float scale = 0.5f;
        float offset = 0.0f;

        if (video_type == "high") {
            alpha = 0.1f;
            scale = 0.8f;
            offset = 0.1f;
        } else if (video_type == "low") {
            alpha = 0.9f;
            scale = 0.15f;
            offset = 0.85f;
        } else { // random
            float pulse = 0.5f + 0.3f * std::sin(2.0f * M_PI * f / source_fps);
            alpha = pulse;
            scale = 0.3f + 0.2f * std::cos(2.0f * M_PI * f / source_fps);
            offset = 0.5f - 0.1f * std::cos(2.0f * M_PI * f / source_fps);
        }

        for (int i = 0; i < width * height; ++i) {
            float val = (1.0f - alpha) * S_blurred[i] + alpha * D_blurred[i];
            val = val * scale + offset;
            int pixel = static_cast<int>(val * 255.0f);
            gray_frame.data[i] = static_cast<uint8_t>(std::min(std::max(pixel, 0), 255));
        }

        // Convert to color BGR frame to ensure absolute compatibility with standard encoders
        cv::cvtColor(gray_frame, color_frame, cv::COLOR_GRAY2BGR);
        writer.write(color_frame);
    }
    writer.release();
}

void processSpeckleVideo(const std::string& filename, int kernel_size, double target_fps, 
                         double exposure_time_ms, int width, int height, int total_frames) {
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file " << filename << std::endl;
        exit(1);
    }

    cv::Mat frame, gray_frame, float_frame;
    std::vector<double> frame_times_ms;
    frame_times_ms.reserve(total_frames);

    int processed_frames = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    while (cap.read(frame)) {
        if (frame.channels() > 1) {
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        } else {
            gray_frame = frame;
        }

        // Convert to 32-bit float normalized to [0.0, 1.0]
        gray_frame.convertTo(float_frame, CV_32FC1, 1.0 / 255.0);

        if (!float_frame.isContinuous()) {
            float_frame = float_frame.clone();
        }

        float* input_ptr = float_frame.ptr<float>();

        auto f_start = std::chrono::high_resolution_clock::now();
        float result = fastSpeckle(input_ptr, height, width, kernel_size);
        auto f_end = std::chrono::high_resolution_clock::now();

        double frame_time = std::chrono::duration<double, std::milli>(f_end - f_start).count();
        frame_times_ms.push_back(frame_time);

        processed_frames++;

        std::cout << "\rProcessing frame: " << processed_frames 
                  << " | Result: " << std::fixed << std::setprecision(5) << result 
                  << " | Time: " << std::setprecision(2) << frame_time << " ms" << std::flush;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_wall_time_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n\n=== Performance Statistics ===" << std::endl;
    if (processed_frames > 0) {
        double min_time = frame_times_ms[0];
        double max_time = frame_times_ms[0];
        double sum_time = 0.0;

        for (double t : frame_times_ms) {
            if (t < min_time) min_time = t;
            if (t > max_time) max_time = t;
            sum_time += t;
        }

        double avg_time_ms = sum_time / processed_frames;
        double processing_fps = 1000.0 / avg_time_ms;
        size_t frame_bytes = width * height * sizeof(float);
        double throughput_gbps = (frame_bytes * processed_frames) / (total_wall_time_sec * 1e9);

        std::cout << "Frames Processed:  " << processed_frames << std::endl;
        std::cout << "Total Execution:   " << std::setprecision(3) << total_wall_time_sec << " s" << std::endl;
        std::cout << "Avg Kernel Time:   " << std::setprecision(3) << avg_time_ms << " ms/frame" << std::endl;
        std::cout << "Min Kernel Time:   " << std::setprecision(3) << min_time << " ms" << std::endl;
        std::cout << "Max Kernel Time:   " << std::setprecision(3) << max_time << " ms" << std::endl;
        std::cout << "Processing Speed:  " << std::setprecision(2) << processing_fps << " FPS" << std::endl;
        std::cout << "Data Throughput:   " << std::setprecision(3) << throughput_gbps << " GB/s" << std::endl;
    } else {
        std::cout << "No frames were processed." << std::endl;
    }
    std::cout << "==============================" << std::endl;
}
#endif

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv) {
    // Help flags or missing argument check
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printHelp(argv[0]);
        return 0;
    }
    if (argc < 3) {
        std::cerr << "Error: Missing required arguments." << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    // Parse parameters
    std::string video_type = argv[1];
    if (video_type != "high" && video_type != "low" && video_type != "random") {
        std::cerr << "Error: Invalid video type '" << video_type << "'. Must be 'high', 'low', or 'random'." << std::endl;
        return 1;
    }

    std::string res_str = argv[2];
    int width = 0, height = 0;
    if (!parseResolution(res_str, width, height)) {
        std::cerr << "Error: Invalid resolution '" << res_str << "'. Correct formats: '1000' or '1024x1024'." << std::endl;
        return 1;
    }

    double source_fps = 80.0;
    if (argc >= 4) {
        try {
            source_fps = std::stod(argv[3]);
            if (source_fps <= 0.0) throw std::exception();
        } catch (...) {
            std::cerr << "Error: Invalid source_fps. Must be a positive number." << std::endl;
            return 1;
        }
    }

    double target_fps = 80.0;
    if (argc >= 5) {
        try {
            target_fps = std::stod(argv[4]);
            if (target_fps <= 0.0) throw std::exception();
        } catch (...) {
            std::cerr << "Error: Invalid target_fps. Must be a positive number." << std::endl;
            return 1;
        }
    }

    int kernel_size = 5;
    if (argc >= 6) {
        try {
            kernel_size = std::stoi(argv[5]);
            if (kernel_size <= 0 || kernel_size % 2 == 0) {
                std::cerr << "Error: Spatial window kernel_size must be a positive odd integer." << std::endl;
                return 1;
            }
        } catch (...) {
            std::cerr << "Error: Invalid kernel_size. Must be an integer." << std::endl;
            return 1;
        }
    }

    double exposure_time_ms = 3.0;
    if (argc >= 7) {
        try {
            exposure_time_ms = std::stod(argv[6]);
            if (exposure_time_ms <= 0.0) throw std::exception();
        } catch (...) {
            std::cerr << "Error: Invalid exposure_time_ms. Must be a positive number." << std::endl;
            return 1;
        }
    }

    int total_frames = static_cast<int>(3.0 * source_fps);

    // Build the output MP4 filename with encoded parameters
    std::string filename = "speckle_" + video_type + "_" 
                         + std::to_string(width) + "x" + std::to_string(height) 
                         + "_s" + formatDouble(source_fps) 
                         + "_t" + formatDouble(target_fps) 
                         + "_k" + std::to_string(kernel_size) 
                         + "_e" + formatDouble(exposure_time_ms) + ".mp4";

    // Print initial parameter configuration block
    std::cout << " - Frame Rate: " << static_cast<int>(source_fps) << " FPS" << std::endl;
    std::cout << " - Frame Count: " << total_frames << " Frames" << std::endl;
    std::cout << " - Target Frame Rate: " << static_cast<int>(target_fps) << " FPS (Period: " 
              << std::fixed << std::setprecision(1) << (1000.0 / target_fps) << " ms)" << std::endl;
    std::cout << " - Spatial LSCI Window: " << kernel_size << "x" << kernel_size << " kernel" << std::endl;
    std::cout << " - Target Exposure (T): " << static_cast<int>(exposure_time_ms) << " ms" << std::endl << std::endl;

    // Print video specifications block
    std::cout << "--- Video Specifications ---" << std::endl;
    std::cout << "Resolution:   " << width << "x" << height << std::endl;
    std::cout << "Total Frames: " << total_frames << std::endl;
    std::cout << "Source FPS:   " << source_fps << std::endl;
    std::cout << "Kernel Size:  " << kernel_size << std::endl;
    std::cout << "Target FPS:   " << target_fps << std::endl;
    std::cout << "----------------------------" << std::endl;

    // Generate or reuse video
    if (fileExists(filename)) {
        std::cout << "Video file '" << filename << "' already exists. Skipping generation." << std::endl << std::endl;
    } else {
        std::cout << "Generating video: " << filename << "..." << std::endl;
        generateSpeckleVideo(filename, video_type, width, height, source_fps, total_frames);
        std::cout << "Video saved successfully!" << std::endl << std::endl;
    }

    // Process and benchmark
    processSpeckleVideo(filename, kernel_size, target_fps, exposure_time_ms, width, height, total_frames);

    return 0;
}
