/*
 * @BEGIN LICENSE
 *
 *  Copyright (c) 2017 Andrew M. James
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 * @END LICENSE
 */

#ifndef reDPD_local_types_h
#define reDPD_local_types_h

#include <cstdio>
#include <utility>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <functional>
#include <memory>
#include <tuple>
#include <sstream>
#include <iterator>
#include <cassert>
#include <stdexcept>

namespace reDPD
{
using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;
using std::vector;
using std::string;
using std::map;
using std::unordered_map;
using std::pair;
using std::function;
using std::stringstream;
using std::ostringstream;
using std::ostream_iterator;
using std::istringstream;

static constexpr double numerical_zero__ = 1.0e-14;

using Dimension = std::vector<size_t>;
using NestedDimension = std::vector<std::vector<size_t>>;

}  // namespace reDPD

#endif
