#pragma once

// Tiny "--key value" / "--flag" command-line parser for the apps.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace vrt {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + a);
      a = a.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        values_[a] = argv[++i];
      } else {
        values_[a] = "";
      }
    }
  }

  bool has(const std::string& k) const { return values_.count(k) != 0; }

  std::string str(const std::string& k, const std::string& def = "") const {
    auto it = values_.find(k);
    return it == values_.end() ? def : it->second;
  }

  std::string required(const std::string& k) const {
    if (!has(k)) throw std::invalid_argument("missing --" + k);
    return values_.at(k);
  }

  long long integer(const std::string& k, long long def) const {
    return has(k) ? std::stoll(values_.at(k)) : def;
  }

  float real(const std::string& k, float def) const {
    return has(k) ? std::stof(values_.at(k)) : def;
  }

 private:
  std::map<std::string, std::string> values_;
};

}  // namespace vrt
