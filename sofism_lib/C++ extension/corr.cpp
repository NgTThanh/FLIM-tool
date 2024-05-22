#include <vector>
#include <algorithm> 
#include <numeric> 
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>  
#include <pybind11/stl.h>  
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>

struct ImageData {
    int detectors;
    int x_pixels;
    int y_pixels;
    int time_series;
    Eigen::MatrixXf data; // Dimensions: (detectors, x_pixels * y_pixels * time_series)

    // Constructor taking a 4D NumPy array as input
    ImageData(int det, int x, int y, int t) : 
    detectors(det), x_pixels(x), y_pixels(y), time_series(t),
    data(detectors, x_pixels * y_pixels * time_series) 
    {
    // Optionally initialize 'data' with zeros or other desired values
        }

    ImageData(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> array)
    {
        // Ensure input is a 4D array
        if (array.ndim() != 4) {
            throw std::runtime_error("Input array must be 4-dimensional");
        }

        // Get array shape information
        auto buf = array.request(); 
        detectors = static_cast<int>(buf.shape[0]);
        x_pixels =  static_cast<int>(buf.shape[1]);
        y_pixels = static_cast<int>(buf.shape[2]);
        time_series = static_cast<int>(buf.shape[3]);

        // Create Eigen Matrix and copy data
        data = Eigen::Map<Eigen::MatrixXf>(static_cast<float*>(buf.ptr), detectors, x_pixels * y_pixels * time_series);
    }

    // Access data using detector, x, y, and time coordinates
    float get_data(int detector, int x, int y, int t) const {
        int index = detector * (x_pixels * y_pixels * time_series) +
                    y * (x_pixels * time_series) +
                    x * time_series + t;
        return data(detector, index);
    }
};

// CorrResult: Assuming a storage for correlation results
struct CorrResult {
    int x_pixels;
    int detectors;
    int y_pixels;
    int tau_range;  
    std::vector<std::vector<std::vector<std::vector<float>>>> results;

    // Constructor
    CorrResult(int x, int d, int y, int tau) : 
        x_pixels(x), detectors(d), y_pixels(y), tau_range(tau)
    {
        results.resize(x_pixels, std::vector<std::vector<std::vector<float>>>(detectors, 
                                        std::vector<std::vector<float>>(detectors, 
                                                std::vector<float>(y_pixels, 0.0f))));
    }

    // Example set_data method 
    void set_data(int X, const std::vector<std::vector<std::vector<float>>>& data) {
        for (int i = 0; i < detectors; ++i) {
            for (int j = 0; j < detectors; ++j) {
                for (int k = 0; k < y_pixels; ++k) {
                    results[X][i][j][k] = data[i][j][k];
                }
            }
        }
    }
};

float mean(const Eigen::VectorXf& data) {
    return data.mean();
}

Eigen::VectorXf get_data_vector(const ImageData& data, int detector, int x, int y) {
    int startIndex = detector * (data.x_pixels * data.y_pixels * data.time_series) + 
                     y * (data.x_pixels * data.time_series) + 
                     x * data.time_series;

    return data.data.block(detector, startIndex, 1, data.time_series);  
}


std::vector<std::vector<std::vector<float>>> process_single_x_cpp(const ImageData &data, int tau_range, int px, bool no_mean) {
    int d = data.detectors;
    int y = data.y_pixels;
    int time_series = data.time_series; 

    // Pre-allocate result matrix using Eigen
    Eigen::MatrixXf res(d, d * y); 

    // Calculate mean vectors efficiently (if needed)
    Eigen::VectorXf means(d);
    if (!no_mean) {
        Eigen::MatrixXf allDetectorsData = data.data.block(0, px * y * time_series, d, y * time_series);
        means = allDetectorsData.colwise().mean();
    }

    // Optimized correlation loop 
    for (int i = 0; i < y; ++i) {
        for (int j = 0; j < d; ++j) {
            Eigen::VectorXf v1 = get_data_vector(data, j, px, i); 

            if (!no_mean) {
                v1 -= means(j) * Eigen::VectorXf::Ones(time_series); // Efficient mean subtraction
            }

            for (int k = 0; k < d; ++k) {
                Eigen::VectorXf v2 = get_data_vector(data, k, px, i);
                if (!no_mean) {
                    v2 -= means(k) * Eigen::VectorXf::Ones(time_series);  
                }

                // Optimized correlation:
                float corr_value = (v1.dot(v2)) / (time_series - 1); 
                res(j, k + i * d) = corr_value;
            }
        }
    }

    // Convert res to the nested vector format
    std::vector<std::vector<std::vector<float>>> nested_res(d, std::vector<std::vector<float>>(d, std::vector<float>(y)));
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < y; ++k) {
                nested_res[i][j][k] = res(i, j + k * d);
            }
        }
    }

    return nested_res;
} 


CorrResult get_corrs_cpp(pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> array, int tau_range, bool no_mean) {
    // Ensure input is a 4D array
    if (array.ndim() != 4) {
        throw std::runtime_error("Input array must be 4-dimensional");
    }

    // Get array shape information
    auto buf = array.request(); 
    int detectors = static_cast<int>(buf.shape[0]);
    int x_pixels =  static_cast<int>(buf.shape[1]);
    int y_pixels = static_cast<int>(buf.shape[2]);
    int time_series = static_cast<int>(buf.shape[3]);

    std::cout << "Input array shape: (" << detectors << ", " << x_pixels << ", " << y_pixels << ", " << time_series << ")" << std::endl;

    // Create ImageData object from the array
    ImageData image_data(array); 

    // Existing logic of get_corrs_cpp 
    int d = image_data.detectors;  // Access from ImageData
    int xs = image_data.x_pixels;
    int y = image_data.y_pixels;

    // Create correlation result storage
    CorrResult results(xs, d, y, tau_range); 

    #pragma omp parallel for 
    for (int X = 0; X < xs; ++X) {
        std::vector<std::vector<std::vector<float>>> partial_res = process_single_x_cpp(image_data, tau_range, X, no_mean); 

        // Update results with proper synchronization
        #pragma omp critical 
        {
            // Directly update CorrResult: Assuming set_data does necessary indexing
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    for (int k = 0; k < y; ++k) {
                        results.results[X][i][j][k] = partial_res[i][j][k]; 
                    }
                }
            }   
        }
    }

    std::cout << "Correlation result shape: (" << xs << ", " << d << ", " << d << ", " << y << ")" << std::endl;

    return results;
}


namespace py = pybind11;

// A wrapper for the ImageData class
PYBIND11_MODULE(correlations, m) {
    py::class_<ImageData>(m, "ImageData")
        // Optionally keep the old constructor
        .def(py::init<int, int, int, int>())  

        // Bind the constructor taking a NumPy array
        .def("__init__", [](ImageData &self, pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> array) {
        new (&self) ImageData(array); 
    }, "Constructor taking a NumPy array")

        .def("get_data", &ImageData::get_data); 
        
    // Binding for CorrResult
    py::class_<CorrResult>(m, "CorrResult")
        .def(py::init<int, int, int, int>())
        // Consider a 'set_data' method if you want modification from Python
        .def("get_data", [](const CorrResult& self, int x, int det1, int det2, int y) {
                return self.results[x][det1][det2][y];
        });

    // Binding for the main correlation function (unchanged)
    m.def("get_corrs", &get_corrs_cpp, "Calculate cross-correlations"); 
    
    // Binding for process_single_x_cpp
    m.def("process_single_x", &process_single_x_cpp, "Process single x");
}


int main(){
    const int detectors = 23;
    const int x_pixels = 21;
    const int y_pixels = 21;
    const int time_series = 3000;
    
    float* data_ptr = new float[detectors * x_pixels * y_pixels * time_series];
    // Fill your data_ptr with random values (as you did with setRandom() before)
    
    // Create a pybind11 array
    pybind11::array_t<float> py_array({detectors, x_pixels, y_pixels, time_series}, data_ptr);
    bool no_mean = true; // Declare and assign the variable 'no_mean'
    
    // Call the function with the variable
    auto results = get_corrs_cpp(py_array, 10, no_mean); 

    // Remember to free the dynamically allocated memory
    delete[] data_ptr;
    return 0;
}
