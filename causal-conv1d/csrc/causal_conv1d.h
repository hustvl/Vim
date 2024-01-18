/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
};

struct ConvParamsBwd: public ConvParamsBase {
    index_t dx_batch_stride;
    index_t dx_c_stride;
    index_t dx_l_stride;
    index_t dweight_c_stride;
    index_t dweight_width_stride;
    index_t dout_batch_stride;
    index_t dout_c_stride;
    index_t dout_l_stride;

    // Common data pointers.
    void *__restrict__ dx_ptr;
    void *__restrict__ dweight_ptr;
    void *__restrict__ dbias_ptr;
    void *__restrict__ dout_ptr;
};

