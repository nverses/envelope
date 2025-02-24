#include <envelope/eigen_types.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace envelope;

static bool exists(const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

// parses csv and loads X and Y double matrix
static std::pair<DMat, DMat> load_csv(const std::string& csvname,
                                      std::vector<int> xidx = {},
                                      std::vector<int> yidx = {})
{
  std::string path = std::string(TEST_DATA_DIR) + "/" + csvname;
  if (!exists(path)) {
    throw std::invalid_argument("test data " + path + " does not exist");
  }
  std::cout << "loading data from: " << path << std::endl;
  // simple if we are just parsing numbers
  std::vector<std::string> head;
  std::vector<std::vector<double>> rows;
  // assums the first line is a header
  std::ifstream is(path);
  std::string line, tok;
  for (int i = 0; std::getline(is, line); i++) {
    std::stringstream ss(line);
    if (i == 0) {
      for (int j = 0; std::getline(ss, tok, ','); j++) {
        head.push_back(tok);
        tok.clear();
      }
    }
    else {
      rows.resize(i);
      for (int j = 0; std::getline(ss, tok, ','); j++) {
        rows[i - 1].push_back(std::atof(tok.c_str()));
        tok.clear();
      }
    }
  }
  // total size
  int nrow = rows.empty() ? 0 : rows.size();
  int ncol = rows.empty() ? 0 : rows[0].size();
  // some flexibility
  if (yidx.empty() && xidx.empty()) {
    for (int j = 0; j < ncol; j++) {
      xidx.push_back(j);
    }
  }
  else if (!yidx.empty() && xidx.empty()) {
    for (int j = 0; j < ncol; j++) {
      // columns that are not in y belongs to x
      if (std::find(yidx.begin(), yidx.end(), j) == yidx.end()) {
        xidx.push_back(j);
      }
    }
  }
  else if (!xidx.empty() && yidx.empty()) {
    for (int j = 0; j < ncol; j++) {
      // columns that are not in x belongs to y
      if (std::find(xidx.begin(), xidx.end(), j) == xidx.end()) {
        yidx.push_back(j);
      }
    }
  }
  DMat x(nrow, xidx.size());
  DMat y(nrow, yidx.size());
  // now slot into double
  int j = 0;
  for (int xj : xidx) {
    for (int i = 0; i < nrow; i++) {
      double val = rows[i][xj];
      x(i, j)    = val;
    }
    j++;
  }
  j = 0;
  for (int yj : yidx) {
    for (int i = 0; i < nrow; i++) {
      double val = rows[i][yj];
      y(i, j)    = val;
    }
    j++;
  }
  return {x, y};
}
