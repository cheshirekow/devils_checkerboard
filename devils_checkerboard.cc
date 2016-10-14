#ifdef __SSE4_1__
#include <nmmintrin.h>
#endif

#include <cassert>
#include <cstdint>
#include <iostream>
#include <list>
#include <set>
#include <vector>

#include <fmt/format.h>
#include <fmt/ostream.h>

// return the value at bit position idx
template <typename NumberType>
inline NumberType Get(NumberType value, NumberType idx) {
  assert(idx < sizeof(NumberType) * 8);
  return (value >> idx) & NumberType(0x01);
}

template <typename NumberType>
struct Setter {
  NumberType* target;
  NumberType index;

  void operator=(NumberType value) {
    // clear the bit position
    *target &= ~(NumberType(0x01) << index);
    // set the bit position
    *target |= (value & NumberType(0x01)) << index;
  }
};

// set the value at bit position idx
template <typename NumberType>
inline Setter<NumberType> Set(NumberType* value, NumberType idx) {
  assert(idx < sizeof(NumberType) * 8);
  return Setter<NumberType>{value, idx};
}

// set the value from a string
template <typename NumberType>
inline void ParseString(const std::string& value_str, NumberType* value) {
  assert(value_str.size() <= sizeof(NumberType) * 8);
  for (NumberType i = 0; i < value_str.size(); ++i) {
    assert(value_str[i] == '0' || value_str[i] == '1');
    Set(value, i) = value_str[i] - '0';
  }
}

template <typename NumberType>
std::string ToString(NumberType value, NumberType ndim) {
  assert(ndim <= sizeof(NumberType) * 8);
  std::string result;
  result.resize(ndim + 1);
  result[0] = 'b';
  for (NumberType i = 0; i < ndim; ++i) {
    result[i + 1] = '0' + Get(value, i);
  }

  return result;
}

// There are 2^n states
template <typename NumberType>
inline NumberType GetNumberOfStates(NumberType ndim) {
  return NumberType(0x01) << ndim;
}

// round number of colors up to next even integer
template <typename NumberType>
inline NumberType GetNumberOfColors(NumberType ndim) {
  // return 2 * ((ndim + 1) / 2);
  return ndim;
}

#ifdef __SSE4_1__
inline uint32_t PopCount(uint32_t value) {
  return _mm_popcnt_u32(value);
}

inline uint64_t PopCount(uint64_t value) {
  return _mm_popcnt_u64(value);
}
#else
// http://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
inline uint32_t PopCount(uint32_t value) {
  // C or C++: use uint32_t
  value = value - ((value >> 1) & 0x55555555);
  value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
  return (((value + (value >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

// http://www.drdobbs.com/parallel/integer-64-bit-optimizations/184405995
inline uint64_t PopCount(uint64_t value) {
  value = (value & 0x5555555555555555LU) + (value >> 1 & 0x5555555555555555LU);
  value = (value & 0x3333333333333333LU) + (value >> 2 & 0x3333333333333333LU);
  value = value + (value >> 4) & 0x0F0F0F0F0F0F0F0FLU;
  value = value + (value >> 8);
  value = value + (value >> 16);
  value = value + (value >> 32) & 0x0000007F;
  return value;
}
#endif

template <typename NumberType>
struct MirrorAssignment {
  MirrorAssignment(NumberType ndim) : ndim_(ndim) {}

  NumberType operator[](NumberType state) const {
    assert(state < (NumberType(0x01) << ndim_));

    NumberType cycle_offset = state % (ndim_ * 2);
    if (cycle_offset < ndim_) {
      return cycle_offset;
    } else {
      return 2 * ndim_ - cycle_offset - 1;
    }
  }

  NumberType ndim_;
};

template <typename NumberType, class ColorAssignment>
bool ValidateColoring(const ColorAssignment& coloring, NumberType ndim) {
  NumberType n_states = GetNumberOfStates<NumberType>(ndim);
  NumberType n_colors = ndim;

  for (NumberType current_state = 0; current_state < n_states;
       ++current_state) {
    // bit-vector of colors seen among neighbors (including self)
    NumberType colors_seen = 0;
    Set(&colors_seen, coloring[current_state]) = 1;

    for (NumberType i = 0; i < ndim; ++i) {
      NumberType neighbor_state = current_state;
      if (Get(current_state, i)) {
        Set(&neighbor_state, i) = 0;
      } else {
        Set(&neighbor_state, i) = 1;
      }

      Set(&colors_seen, coloring[neighbor_state]) = 1;
    }

    if (PopCount(colors_seen) != n_colors) {
      fmt::print(
          std::cout,
          "For state {1:0{0}b}, saw {2} ({3:0{0}b}) colors, expected {4:d}\n",
          ndim, current_state, PopCount(colors_seen), colors_seen, n_colors);
            fmt::print(std::cout, "colors_seen: {:08b}\n", colors_seen);
      return false;
    }
  }

  return true;
}

static const std::string kFormat2 =
    "\
  (10) o ----- o (11)   (00) : {0:d} \n\
       |       |        (01) : {1:d} \n\
       |       |        (10) : {2:d} \n\
  (00) o-------o (01)   (11) : {3:d} \n\
";

static const std::string kFormat3 =
    "\
\n\
    (110) o-------o (111)   (000) : {0:d} \n\
         /|      /|         (001) : {1:d} \n\
 (010)  / |     / |         (010) : {2:d} \n\
       o ----- o  o (101)   (011) : {3:d} \n\
       | /     | /          (100) : {4:d} \n\
       |/      |/           (101) : {5:d} \n\
 (000) o-------o (001)      (110) : {6:d} \n\
                            (111) : {7:d} \n\
";

template <typename NumberType, class ColorAssignment>
void PrintColoring(std::ostream& out, const ColorAssignment& coloring,
                   NumberType ndim) {
  switch (ndim) {
    case 2: {
      fmt::print(out, kFormat2, coloring[0], coloring[1], coloring[2],
                 coloring[3]);
      break;
    }

    case 3: {
      fmt::print(out, kFormat3, coloring[0], coloring[1], coloring[2],
                 coloring[3], coloring[4], coloring[5], coloring[6],
                 coloring[7]);
      break;
    }

    case 4: {
      break;
    }

    default:
      out << "No visualization for dimension " << ndim << "\n";
  }
}

int main(int argc, char** argv) {
  for (uint64_t ndim = 2; ndim < 17; ndim *= ndim) {
    MirrorAssignment<uint64_t> coloring(ndim);
    fmt::print(std::cout, "\n\nn = {}, {} states, {} colors\n", ndim,
               GetNumberOfStates<uint64_t>(ndim),
               GetNumberOfColors<uint64_t>(ndim));

    PrintColoring(std::cout, coloring, ndim);
    bool is_valid = ValidateColoring(coloring, ndim);
    fmt::print(std::cout, "Validated: {}\n", (is_valid ? "yes" : "no"));
    std::cout.flush();
  }
  return 0;
}

template <typename NumberType>
struct TopologicalCompare {
  bool operator()(NumberType a, NumberType b) {
    // Same depth, return sorted order
    if (PopCount(a) == PopCount(b)) {
      return a > b;
      // Different depth, handle lower depth first
    } else {
      return PopCount(a) < PopCount(b);
    }
  }
};

// Cycle over colors in topological order
template <typename NumberType>
std::vector<NumberType> GenerateColoring(NumberType ndim) {
  NumberType n_states = GetNumberOfStates<NumberType>(ndim);
  NumberType num_colors = GetNumberOfColors<NumberType>(ndim);
  NumberType next_color = 0;

  std::set<NumberType, TopologicalCompare<NumberType>> queue;

  std::vector<NumberType> result(n_states);
  std::vector<bool> open(n_states, false);
  std::vector<bool> closed(n_states, false);

  open[0] = true;
  queue.insert(0);

  while (queue.size() > 0) {
    auto front_iter = queue.begin();
    NumberType current_state = *front_iter;
    queue.erase(front_iter);
    assert(current_state < result.size());

    if (!closed[current_state]) {
      closed[current_state] = true;
      result[current_state] = next_color;
      next_color = (next_color + 1) % num_colors;
    }

    for (NumberType i = 0; i < ndim; ++i) {
      NumberType child_state = current_state;
      if (Get(child_state, i) == 0) {
        Set(&child_state, i) = 1;
        if (!open[child_state]) {
          open[child_state] = true;
          queue.insert(child_state);
        }
      }
    }
  }

  return result;
}

int old_main(int argc, char** argv) {
  for (uint32_t ndim = 2; ndim < 5; ++ndim) {
    fmt::print(std::cout, "\n\nn = {}, {} states, {} colors\n", ndim,
               GetNumberOfStates<uint32_t>(ndim),
               GetNumberOfColors<uint32_t>(ndim));

    std::vector<uint32_t> coloring = GenerateColoring<uint32_t>(ndim);
    PrintColoring(std::cout, coloring, ndim);
    bool is_valid = ValidateColoring(coloring, ndim);
    fmt::print(std::cout, "Validated: {}\n", (is_valid ? "yes" : "no"));
    std::cout.flush();
  }
  return 0;
}