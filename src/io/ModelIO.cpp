#include "ml/io/ModelIO.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace ml::io {

void write_header(std::ostream& os, std::string_view model_name, int version) {
    os << "# QuarkML " << model_name << " v" << version << "\n";
}

void write_kv(std::ostream& os, std::string_view key, std::string_view value) {
    os << key << '=' << value << '\n';
}

void write_vec(std::ostream& os, std::string_view key, const std::vector<double>& v, int precision) {
    os << key << '=';
    os << std::setprecision(precision) << std::fixed;
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ',';
    }
    os << '\n';
}

static inline bool is_blank_or_comment(const std::string& line) {
    if (line.empty()) return true;
    // ignore spaces prior to '#'
    size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    return (i >= line.size()) || (line[i] == '#');
}

std::unordered_map<std::string, std::string> parse_kv_file(std::istream& is) {
    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(is, line)) {
        if (is_blank_or_comment(line)) continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            // tolerate lines without '=' by skipping
            continue;
        }
        std::string key = trim(line.substr(0, pos));
        std::string val = trim(line.substr(pos + 1));
        kv[std::move(key)] = std::move(val);
    }
    return kv;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, delim)) {
        parts.emplace_back(cur);
    }
    // Handle trailing delimiter (e.g., "1,2,")
    if (!s.empty() && s.back() == delim) parts.emplace_back("");
    return parts;
}

std::string trim(std::string s) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::vector<double> parse_vec(const std::string& csv) {
    std::vector<double> out;
    for (auto& token : split(csv, ',')) {
        auto t = trim(token);
        if (t.empty()) {
            out.emplace_back(0.0);
            continue;
        }
        // robust stod with locale-independence
        try {
            size_t idx = 0;
            double val = std::stod(t, &idx);
            if (idx != t.size()) {
                throw std::invalid_argument("Trailing characters in number: " + t);
            }
            out.emplace_back(val);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("ModelIO::parse_vec failed on token '") + t + "': " + e.what());
        }
    }
    return out;
}

} // namespace ml::io
