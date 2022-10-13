#include <vector>
#include <string>

// Device options
struct DeviceCPU;
struct DeviceGPU;
using DefaultDevice = DeviceCPU;

// State options
struct StatePhys;
struct StateCoeff;
using DefaultState = StatePhys;

struct CollectionInfo {
    size_t offset;
};

struct VariableInfo {
    std::string name;
};

template<typename TState = DefaultState, typename TType = double>
class Field {
    public:
        Field();
        Field(const Field&) = default;
        ~Field();

        bool IsOnDevice() {
            return on_device;
        }

    private:
        TType* data_h = nullptr;  // host
        TType* data_d = nullptr;  // device
        bool on_device = false;   // true = data is on the device

        std::vector<CollectionInfo> coll_info;
        size_t nvar;
        std::vector<VariableInfo> var_info;

        void HostToDevice() {

        }

        void DeviceToHost() {

        }

};
