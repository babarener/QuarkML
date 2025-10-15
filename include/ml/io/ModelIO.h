#pragma once
#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ml::io {

/// Write a human-readable header like: "# QuarkML LinearRegression v1"
void write_header(std::ostream& os, std::string_view model_name, int version);

/// Write a line "key=value\n"
void write_kv(std::ostream& os, std::string_view key, std::string_view value);

/// Write a CSV vector as "key=v1,v2,v3\n" (fixed precision handled by caller if needed)
void write_vec(std::ostream& os, std::string_view key, const std::vector<double>& v, int precision = 10);

/// Parse a key=value text file into a dictionary. Lines starting with '#' or blank are ignored.
/// If a key repeats, the last value wins.
std::unordered_map<std::string, std::string> parse_kv_file(std::istream& is);

/// Parse a comma-separated list of doubles into a vector.
std::vector<double> parse_vec(const std::string& csv);

/// Small utilities (exposed for reuse)
std::string trim(std::string s);
std::vector<std::string> split(const std::string& s, char delim);

} // namespace ml::io
