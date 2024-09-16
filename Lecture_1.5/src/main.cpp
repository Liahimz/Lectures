#include <Python.h>
#include <opencv2/opencv.hpp>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <config.h>

// Initialize the NumPy C API
// Initialize the NumPy C API
int init_numpy() {
    if (_import_array() < 0) {  // Directly use _import_array to handle the return value
        PyErr_Print();  // Print the error if NumPy initialization fails
        return -1;  // Return -1 to indicate an error
    }
    return 0;  // Return 0 to indicate success
}

// Function to load the Python module and perform inference
std::vector<std::vector<float>> run_inference(const std::string& image_path) {
    // Initialize the Python interpreter
    Py_Initialize();
    // Initialize NumPy
    if (init_numpy() != 0) {
        std::cout << "Failed to initialize NumPy." << std::endl;
        return {};  // Return an empty vector if NumPy initialization fails
    }

    // Redirect Python stdout and stderr to C++ standard output
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.stdout = sys.stderr = open('/dev/stdout', 'w')");
    
    // Add the directory containing the 'model_inference' module to the Python path
    PyObject* sys_path = PySys_GetObject("path");
    PyList_Append(sys_path, PyUnicode_FromString(LD_PATH));  // .so file path

    PyObject* pName = PyUnicode_DecodeFSDefault("model_inference");  // Module name
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule) {
    // Force reload the module
        PyObject* pReloadedModule = PyImport_ReloadModule(pModule);
        Py_DECREF(pModule);
        pModule = pReloadedModule;
    }
    
    std::vector<std::vector<float>> bounding_boxes;
    if (pModule) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "inference");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(image_path.c_str()));
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue) {
                // Assuming the predictions are a numpy array, we convert it to C++ data types
                PyArrayObject* np_arr = reinterpret_cast<PyArrayObject*>(pValue);
                int rows = PyArray_SHAPE(np_arr)[0];
                int cols = PyArray_SHAPE(np_arr)[1];
                float* data = static_cast<float*>(PyArray_DATA(np_arr));
                
                for (int i = 0; i < rows; ++i) {
                    std::vector<float> bbox(cols);
                    for (int j = 0; j < cols; ++j) {
                        bbox[j] = data[i * cols + j];
                    }
                    bounding_boxes.push_back(bbox);
                }
                Py_DECREF(pValue);
            }
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }

    // Finalize the Python interpreter
    Py_Finalize();

    return bounding_boxes;
}

// Function to draw bounding boxes and save the image
void draw_bounding_boxes(const std::string& image_path, const std::vector<std::vector<float>>& bboxes, const std::string& output_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cout << "Failed to load image: " << image_path << std::endl;
        return;
    }

    for (const auto& bbox : bboxes) {
        cv::Point pt1(bbox[0], bbox[1]);
        cv::Point pt2(bbox[2], bbox[3]);
        cv::rectangle(image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite(output_path, image);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return 1;
    }

    std::string input_image = argv[1];
    std::string output_image = argv[2];

    // Run inference and get bounding boxes
    std::vector<std::vector<float>> bboxes = run_inference(input_image);

    // Draw bounding boxes on the image and save it
    draw_bounding_boxes(input_image, bboxes, output_image);

    return 0;
}
