/*
 * @BEGIN LICENSE
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

#ifndef reDPD_tensorimpl_h
#define reDPD_tensorimpl_h
#include "libxm/xm.h"
#include "local_types.h"
namespace reDPD
{
void init(const std::string &scratch_loc);
void finalize();

class XMTensor
{
 private:
  struct xm_tensor_t_del {
    void operator()(xm_tensor_t *ptr)
    {
      xm_tensor_free_block_data(ptr);
      xm_tensor_free(ptr);
    }
  };
  struct xm_block_space_t_del {
    void operator()(xm_block_space_t *ptr) { xm_block_space_free(ptr); }
  };
  struct xm_allocator_t_del {
    void operator()(xm_allocator_t *ptr) { xm_allocator_destroy(ptr); }
  };

 public:
  using inner_tensor_t = unique_ptr<xm_tensor_t, xm_tensor_t_del>;
  using bs_t = unique_ptr<xm_block_space_t, xm_block_space_t_del>;
  using allocator_t = unique_ptr<xm_allocator_t, xm_allocator_t_del>;
  struct BlockData {
    vector<size_t> abs_start;
    vector<size_t> abs_end;
    vector<size_t> shape;
    vector<size_t> index;
    size_t nele;
    enum block_type {
      zero = XM_BLOCK_TYPE_ZERO,
      canonical = XM_BLOCK_TYPE_CANONICAL,
      derivative = XM_BLOCK_TYPE_CANONICAL
    };
    block_type type;
    BlockData() : abs_start(0), abs_end(0), shape(0), index(0), nele(0), type(BlockData::zero) {}
    BlockData(const vector<size_t> &starts, const vector<size_t> &ends, const vector<size_t> &idx)
        : abs_start(starts), abs_end(ends), index(idx)
    {
      shape.resize(abs_start.size());
      for (size_t i = 0; i < abs_start.size(); i++) {
        shape[i] = abs_end[i] - abs_start[i];
      }
      nele = 1;
      for (auto &len : shape) {
        nele *= len;
      }
      // For now we only have zero blocks and canonical blocks
      if (nele > 0) {
        type = canonical;
      } else {
        type = zero;
      }
    }
  };

 public:
  XMTensor()
      : rank_(0),
        my_irrep_(0),
        nirrep_(0),
        dims_(0, vector<size_t>(0, 0)),
        sym_allowed_blocks_(0, vector<size_t>(0)),
        blocks_(0),
        tensor_(nullptr)
  {
  }
  XMTensor(const vector<vector<size_t>> &dims, size_t my_irrep = 0);

  // factory function
  static XMTensor build(const vector<vector<size_t>> &dims, size_t my_irrep = 0);

  // factory init like this function
  static XMTensor build_like(const XMTensor &other);

  size_t rank() const { return rank_; }

  size_t my_irrep() const { return my_irrep_; }

  size_t nirrep() const { return nirrep_; }

  size_t disk_footprint() const { return 0; }

  vector<vector<size_t>> dims() const { return dims_; }

  void block_iterate(const function<void(const BlockData &)> &func);
  void set_block_data(const vector<size_t> &block_idx, vector<double> block_data,
                      bool stride_one_left = true);

  void set_block_data(const vector<size_t> &block_idx, double *block_data,
                      bool stride_one_left = true);

  void block_fill_iterate(const function<std::vector<double>(const BlockData &)> &func,
                          bool stride_one_left = true);

  void fill(vector<vector<double>> data, bool stride_one_left = true);

  static void set_scratch_path(const string &path);

  // => Operational Engines <= //
  /*
   * Note I plan to hide these from the general user eventually.
   * This will allow me to modify the implementation of the operations
   * if deemed necessary without breaking interface. I want to force the
   * general library user to go through the expressions interface.
   */
  // Addition
  static void add(double pre_left, XMTensor &left_tensor, const string &left_idx, double pre_right,
                  const XMTensor &right_tensor, const string &right_idx);
  // Copy
  static void data_copy(XMTensor &to, const string &idx_to, double alpha, const XMTensor &from,
                        const string &idx_from);
  // Contraction
  static void contract(double alpha, const XMTensor &A, const string &a_idx, const XMTensor &B,
                       const string &b_idx, double beta, XMTensor &C, const string &c_idx);
  // Data Equal
  static bool data_equal(const XMTensor &lhs, const XMTensor &rhs);

  vector<size_t> get_block_dims(const vector<size_t> &block_idx) const;
  vector<double> get_block_data(const BlockData &blk_info) const;
  // Dimension Equal

  // For debug
  double get_element(const vector<size_t> &block_idx, const vector<size_t> &elm_idx) const;

 protected:
  size_t rank_;
  size_t my_irrep_;
  size_t nirrep_;
  vector<vector<size_t>> dims_;
  inner_tensor_t tensor_;
  vector<size_t> abs_dims_;
  vector<vector<size_t>> sym_allowed_blocks_;
  vector<BlockData> blocks_;

  // Member functions
 protected:
  vector<vector<size_t>> blocks_by_symmetry() const;
  void init_inner_tensor(const bs_t &block_space);

  // Static Member functions
  static xm_dim_t make_xm_dim(const vector<size_t> &vector_dim);

  // Static Data Members
  static string scratch_path__;
  static allocator_t alloc__;
};
}  // namespace reDPD
#endif
