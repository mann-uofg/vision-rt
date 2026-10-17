#include "vrt/stats.h"

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace vrt {

double percentile(std::vector<double> v, double q) {
  if (v.empty()) throw std::invalid_argument("percentile of empty vector");
  std::sort(v.begin(), v.end());
  const double pos = q / 100.0 * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

double process_cpu_ms() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  auto ms = [](const timeval& tv) { return tv.tv_sec * 1e3 + tv.tv_usec / 1e3; };
  return ms(u.ru_utime) + ms(u.ru_stime);
}

Summary summarize(const std::vector<double>& v) {
  Summary s;
  s.n = v.size();
  if (v.empty()) return s;
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  s.p50 = percentile(v, 50);
  s.p90 = percentile(v, 90);
  s.p99 = percentile(v, 99);
  s.max = *std::max_element(v.begin(), v.end());
  return s;
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

void JsonWriter::key(const std::string& k) {
  if (!first_.back()) out_ += ",";
  first_.back() = false;
  out_ += "\n" + std::string(static_cast<std::size_t>(2 * depth_), ' ');
  if (!k.empty()) out_ += "\"" + json_escape(k) + "\": ";
}

JsonWriter& JsonWriter::begin_object(const std::string& k) {
  if (depth_ > 0) key(k);
  out_ += "{";
  first_.push_back(true);
  ++depth_;
  return *this;
}

JsonWriter& JsonWriter::end_object() {
  --depth_;
  first_.pop_back();
  out_ += "\n" + std::string(static_cast<std::size_t>(2 * depth_), ' ') + "}";
  return *this;
}

JsonWriter& JsonWriter::field(const std::string& k, const std::string& v) {
  key(k);
  out_ += "\"" + json_escape(v) + "\"";
  return *this;
}

JsonWriter& JsonWriter::field(const std::string& k, const char* v) {
  return field(k, std::string(v));
}

JsonWriter& JsonWriter::field(const std::string& k, double v) {
  key(k);
  if (!std::isfinite(v)) {
    out_ += "null";
  } else {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    out_ += buf;
  }
  return *this;
}

JsonWriter& JsonWriter::field(const std::string& k, long long v) {
  key(k);
  out_ += std::to_string(v);
  return *this;
}

JsonWriter& JsonWriter::field(const std::string& k, int v) {
  return field(k, static_cast<long long>(v));
}

JsonWriter& JsonWriter::field(const std::string& k, bool v) {
  key(k);
  out_ += v ? "true" : "false";
  return *this;
}

JsonWriter& JsonWriter::field(const std::string& k, const Summary& s) {
  begin_object(k);
  field("n", static_cast<long long>(s.n));
  field("mean", s.mean).field("p50", s.p50).field("p90", s.p90).field("p99", s.p99);
  field("max", s.max);
  return end_object();
}

std::string JsonWriter::str() const { return out_ + "\n"; }

}  // namespace vrt
