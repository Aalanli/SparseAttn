#include <cuda_fp16.h>
#include <string>

namespace type_repr {

template <typename T>
std::string to_std_type_str() {
    return std::string(typeid(T).name());
}

template <>
std::string to_std_type_str<double>() {
    return std::string("d");
}
template <>
std::string to_std_type_str<float>() {
    return std::string("f");
}
template <>
std::string to_std_type_str<int>() {
    return std::string("i");
}
template <>
std::string to_std_type_str<long long>() {
    return std::string("l");
}
template <>
std::string to_std_type_str<unsigned int>() {
    return std::string("ui");
}
template <>
std::string to_std_type_str<float4>() {
    return std::string("f4");
}
template <>
std::string to_std_type_str<float2>() {
    return std::string("f2");
}
template <>
std::string to_std_type_str<float1>() {
    return std::string("f1");
}
template <>
std::string to_std_type_str<half>() {
    return std::string("h");
}
template <>
std::string to_std_type_str<half2>() {
    return std::string("h2");
}

}