#pragma once

#include <string>
#include <vector>

namespace vrt {

struct Summary {
  std::size_t n = 0;
  double mean = 0, p50 = 0, p90 = 0, p99 = 0, max = 0;
};

// Percentiles with linear interpolation between closest ranks (numpy's default), so C++ and
// Python results files are directly comparable.
double percentile(std::vector<double> v, double q);
Summary summarize(const std::vector<double>& v);

// Minimal JSON object writer: enough for flat-ish results files, no dependency.
class JsonWriter {
 public:
  JsonWriter& begin_object(const std::string& key = "");
  JsonWriter& end_object();
  JsonWriter& field(const std::string& key, const std::string& value);
  JsonWriter& field(const std::string& key, const char* value);
  JsonWriter& field(const std::string& key, double value);
  JsonWriter& field(const std::string& key, long long value);
  JsonWriter& field(const std::string& key, int value);
  JsonWriter& field(const std::string& key, bool value);
  JsonWriter& field(const std::string& key, const Summary& s);
  std::string str() const;

 private:
  void key(const std::string& k);
  std::string out_;
  std::vector<bool> first_{true};
  int depth_ = 0;
};

std::string json_escape(const std::string& s);

}  // namespace vrt
