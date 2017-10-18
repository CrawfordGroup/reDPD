/*
 * @BEGIN LICENSE
 *
 *  MIT License
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

#include "xmtensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
namespace reDPD
{
string XMTensor::scratch_path__ = "xmpagefile";

XMTensor::allocator_t XMTensor::alloc__ =
    XMTensor::allocator_t(nullptr, XMTensor::xm_allocator_t_del());

void init(const std::string &scratch_loc) { XMTensor::set_scratch_path(scratch_loc); }

void finalize() { return; }

XMTensor XMTensor::build(const vector<vector<size_t>> &dims, size_t my_irrep)
{
  XMTensor R(dims, my_irrep);
  return R;
}

XMTensor XMTensor::build_like(const XMTensor &other)
{
  XMTensor R(other.dims(), other.my_irrep());
  return R;
}

void XMTensor::set_scratch_path(const std::string &scratch_loc) { scratch_path__ = scratch_loc; }

XMTensor::XMTensor(const vector<vector<size_t>> &dims, size_t my_irrep)
    : dims_(dims), my_irrep_(my_irrep), tensor_(nullptr, XMTensor::xm_tensor_t_del())
{
  rank_ = dims.size();
  nirrep_ = dims[0].size();
  for (auto &subdim : dims) {
    assert(subdim.size() == nirrep_);
  }
  // symmetry
  sym_allowed_blocks_ = blocks_by_symmetry();
  abs_dims_ = vector<size_t>(rank_);
  for (size_t nr = 0; nr < rank_; nr++) {
    abs_dims_[nr] = 0;
    for (auto &dim_nr : dims_[nr]) {
      abs_dims_[nr] += dim_nr;
    }
  }
  bs_t tensor_bs(xm_block_space_create(make_xm_dim(abs_dims_)), xm_block_space_t_del());
  for (size_t rank_n = 0; rank_n < rank_; rank_n++) {
    size_t split_at = 0;
    for (size_t split_n = 0; split_n < nirrep_ - 1; split_n++) {
      split_at += dims_[rank_n][split_n];
      xm_block_space_split(tensor_bs.get(), rank_n, split_at);
    }
  }

  for (auto &block_idx_vector : sym_allowed_blocks_) {
    vector<size_t> starts(rank_, 0);
    vector<size_t> ends(rank_, 0);
    for (size_t i = 0; i < rank_; i++) {
      size_t irr_this_rank = block_idx_vector[i];
      for (size_t h = 0; h < irr_this_rank; h++) {
        starts[i] += dims_[i][h];
      }
      ends[i] = starts[i] + dims_[i][irr_this_rank];
    }
    blocks_.push_back(BlockData(starts, ends, block_idx_vector));
  }

  init_inner_tensor(tensor_bs);

  for (auto &block_info : blocks_) {
    if (block_info.type == BlockData::canonical) {
      xm_tensor_set_canonical_block(tensor_.get(), make_xm_dim(block_info.index));
    }
  }
  // tensor_bs goes out of scope and the deleter takes care of cleanup
}

void XMTensor::init_inner_tensor(XMTensor::bs_t const &block_space)
{
  if (!alloc__) {
    if (scratch_path__ == "") {
      alloc__.reset(xm_allocator_create(NULL));
    } else {
      alloc__.reset(xm_allocator_create(scratch_path__.c_str()));
    }
  }
  tensor_.reset(xm_tensor_create(block_space.get(), XM_SCALAR_DOUBLE, alloc__.get()));
}

xm_dim_t XMTensor::make_xm_dim(const vector<size_t> &vector_dim)
{
  assert(vector_dim.size() <= XM_MAX_DIM);
  xm_dim_t ret_dim;
  ret_dim.n = vector_dim.size();
  for (size_t a = 0; a < ret_dim.n; a++) {
    ret_dim.i[a] = vector_dim[a];
  }
  return ret_dim;
}

vector<vector<size_t>> XMTensor::blocks_by_symmetry() const
{
  auto get_max_irrep_tuple = [](size_t base, size_t exp) {
    size_t result = 1;
    while (exp) {
      if (exp & 1) {
        result *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return result;
  };
  vector<vector<size_t>> allowed_irrep_combos;
  size_t max_n_tup = get_max_irrep_tuple(nirrep_, rank_);
  for (size_t dp_cnt = 0; dp_cnt < max_n_tup; dp_cnt++) {
    vector<size_t> irrep_tup(rank_);
    size_t current_n = dp_cnt;
    for (size_t i = 0; i < rank_; i++) {
      size_t gamma = current_n % nirrep_;
      irrep_tup[rank_ - i - 1] = gamma;
      current_n /= nirrep_;
    }
    size_t dp_all_idx = my_irrep_;
    for (auto &gamma_h : irrep_tup) {
      dp_all_idx = dp_all_idx ^ gamma_h;
    }
    if (dp_all_idx == 0) {
      allowed_irrep_combos.push_back(irrep_tup);
    }
  }
  std::sort(allowed_irrep_combos.begin(), allowed_irrep_combos.end(),
            [](const std::vector<size_t> &a, const std::vector<size_t> &b) {
              size_t sub = 0;
              for (; sub < a.size(); sub++) {
                if (a[sub] < b[sub])
                  continue;
                else
                  break;
              }
              return a[sub] < b[sub];
            });
  return allowed_irrep_combos;
}

vector<size_t> XMTensor::get_block_dims(const vector<size_t> &block_idx) const
{
  vector<size_t> block_dims(rank_);
  for (size_t i = 0; i < rank_; i++) {
    block_dims[i] = dims_[i][block_idx[i]];
  }
  return block_dims;
}

void XMTensor::set_block_data(const vector<size_t> &block_idx, double *block_data,
                              bool stride_one_left)
{
  xm_dim_t xm_block_idx = make_xm_dim(block_idx);
  if (stride_one_left) {
    assert(xm_tensor_get_block_type(tensor_.get(), xm_block_idx) == XM_BLOCK_TYPE_CANONICAL);
    xm_tensor_write_block(tensor_.get(), xm_block_idx, block_data);
  } else {
    // We have to to the explicit transpose if passed a buffer that has unit stride in the rightmost
    // index instead of the leftmost index (libxm assumes leftmost index is unit stride)
    size_t nele = xm_tensor_get_block_size(tensor_.get(), xm_block_idx);
    // Get the block shape to increment through
    xm_dim_t b_shape = xm_tensor_get_block_dims(tensor_.get(), xm_block_idx);
    // Reverse it for rowmaj shaape
    xm_dim_t r_b_shape;
    r_b_shape.n = b_shape.n;
    for (size_t i = 0; i < rank_; i++) {
      r_b_shape.i[rank_ - 1 - i] = b_shape.i[i];
    }
    std::reverse_copy(b_shape.i, b_shape.i + rank_, r_b_shape.i);
    // Col major counter
    size_t cm_pos = 0;
    // Index for rm order
    xm_dim_t src_idx;
    src_idx.n = rank_;
    double *tmp = new double[nele];
    for (xm_dim_t tgt_idx = xm_dim_zero(rank_); xm_dim_less(&tgt_idx, &b_shape);
         xm_dim_inc(&tgt_idx, &b_shape)) {
      for (size_t i = 0; i < rank_; i++) {
        src_idx.i[rank_ - 1 - i] = tgt_idx.i[i];
      }
      size_t rm_pos = xm_dim_offset(&src_idx, &r_b_shape);
      tmp[cm_pos] = block_data[rm_pos];
      cm_pos++;
    }
    xm_tensor_write_block(tensor_.get(), xm_block_idx, tmp);
    delete[] tmp;
  }
}

void XMTensor::set_block_data(const vector<size_t> &block_idx, vector<double> block_data,
                              bool stride_one_left)
{
  set_block_data(block_idx, block_data.data(), stride_one_left);
}

void XMTensor::block_iterate(const function<void(const BlockData &)> &func)
{
  for (const auto &blk_info : blocks_) {
    func(blk_info);
  }
}

void XMTensor::block_fill_iterate(const function<std::vector<double>(const BlockData &)> &func,
                                  bool stride_one_left)
{
  for (const auto &blk_info : blocks_) {
    set_block_data(blk_info.index, func(blk_info), stride_one_left);
  }
}

void XMTensor::fill(vector<vector<double>> data, bool stride_one_left)
{
  auto dat_it = data.begin();
  for (const auto &blk_info : blocks_) {
    vector<double> block_data(*dat_it);
    set_block_data(blk_info.index, block_data, stride_one_left);
    dat_it++;
  }
}

void XMTensor::add(double pre_left, XMTensor &left_tensor, const string &left_idx, double pre_right,
                   const XMTensor &right_tensor, const string &right_idx)
{
  xm_add(pre_left, left_tensor.tensor_.get(), pre_right, right_tensor.tensor_.get(),
         left_idx.c_str(), right_idx.c_str());
}

void XMTensor::data_copy(XMTensor &to, const string &idx_to, double alpha, const XMTensor &from,
                         const string &idx_from)
{
  xm_copy(to.tensor_.get(), alpha, from.tensor_.get(), idx_to.c_str(), idx_from.c_str());
}

void XMTensor::contract(double alpha, const XMTensor &A, const string &a_idx, const XMTensor &B,
                        const string &b_idx, double beta, XMTensor &C, const string &c_idx)
{
  xm_contract(alpha, A.tensor_.get(), B.tensor_.get(), beta, C.tensor_.get(), a_idx.c_str(),
              b_idx.c_str(), c_idx.c_str());
}

bool XMTensor::data_equal(const XMTensor &lhs, const XMTensor &rhs)
{
  if (lhs.blocks_.size() != rhs.blocks_.size()) {
    std::cout << "lhs.nblock != rhs.nblocks\n";
    return false;
  } else {
    double sum_delta = 0.0;
    for (size_t n = 0; n < lhs.blocks_.size(); n++) {
      if (rhs.blocks_[n].nele != lhs.blocks_[n].nele) {
        return false;
      } else {
        vector<double> lhs_data(lhs.blocks_[n].nele);
        vector<double> rhs_data(rhs.blocks_[n].nele);
        xm_tensor_read_block(rhs.tensor_.get(), make_xm_dim(rhs.blocks_[n].index), rhs_data.data());
        xm_tensor_read_block(lhs.tensor_.get(), make_xm_dim(lhs.blocks_[n].index), lhs_data.data());
        bool all_data_equal = std::equal(rhs_data.begin(), rhs_data.end(), lhs_data.begin(),
                                         [&sum_delta](double a, double b) {
                                           double delta = fabs(a - b);
                                           if (delta < (numerical_zero__)) {
                                             return true;
                                           } else {
                                             sum_delta += delta;
                                             return false;
                                           }
                                         });
        if (not all_data_equal) {
          std::cout << "\t sum of diff block[" << n << "]= ";
          std::cout.precision(15);
          std::cout << sum_delta;
          return false;
        } else {
          continue;
        }
      }
    }
  }
  return true;
}

// For testing only
double XMTensor::get_element(const vector<size_t> &block_idx, const vector<size_t> &elm_idx) const
{
  vector<size_t> abs_idx;
  // loop over ranks
  for (size_t i = 0; i < rank_; i++) {
    size_t offset_this_rank = 0;
    // loop over blockdim for this rank up to the current block idx
    for (size_t n = 0; n < block_idx[i]; n++) {
      offset_this_rank += dims_[i][n];
    }
    // add the block offset to the element offset within the block
    abs_idx.push_back(offset_this_rank + elm_idx[i]);
  }
  xm_scalar_t retval = xm_tensor_get_element(tensor_.get(), make_xm_dim(abs_idx));
  return retval.real();
}

// For testing only
vector<double> XMTensor::get_block_data(const BlockData &blk_info) const
{
  assert(blk_info.type == BlockData::canonical);
  vector<double> ret(blk_info.nele, 0.0);
  double *ret_ptr = ret.data();
  xm_tensor_read_block(tensor_.get(), make_xm_dim(blk_info.index), ret_ptr);
  return ret;
}

}  // namespace reDPD
